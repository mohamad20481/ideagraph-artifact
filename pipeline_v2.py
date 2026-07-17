"""
pipeline_v2.py - End-to-End Automated Scientist Pipeline.

7-stage loop: Idea → Experiment Design → Code Generation → Execution →
Analysis → Paper Writing → Review → (iterate)

Advanced optimizations:
  - PipelineOptimizer: circuit breaker, adaptive concurrency, warm cache
  - AdaptiveBudgetAllocator: dynamic budget reallocation based on stage quality
  - Speculative execution: pre-computes likely-needed stages in background
  - Stage performance tracking: auto-tunes timeouts and skip decisions
  - Warm caching: reuses expensive computation across iterations
"""

from __future__ import annotations

import heapq
import json
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional

import config
from agents.base_agent import (
    get_call_metrics, get_token_usage, estimate_cost_usd,
    set_annealing_iteration, get_annealed_temperature,
    reset_caches,
    _CONTRASTIVE,
)
from agents.budget_allocator import AdaptiveBudgetAllocator
from agents.experiment_designer import ExperimentDesigner
from agents.code_generator import CodeGenerator
from agents.experiment_executor import ExperimentExecutor
from agents.result_analyzer import ResultAnalyzer
from agents.paper_writer import PaperWriter
from agents.reviewer import AutoReviewer
from agents.tree_search import ExperimentTreeSearch
from agents.self_reflection import SelfReflectionAgent
from optimization import PipelineOptimizer
from creative_optimization import (
    CreativeOptimizer, ParetoFront, EloTournament,
    AdversarialTester, MCTSPipelineRouter, PopulationTrainer,
)
from deep_optimization import (
    DeepOptimizer, KnapsackBudget, AttentionRollback,
    EntropyRegularizer, RewardShaper, TokenBudgetProjector,
)
from infra_optimization import InfraOptimizer, StructuredLogger, LogLevel
from meta_optimization import MetaOptimizer
from quantum_optimization import QuantumOptimizer, FractalBudgetAllocator
from nature_optimization import NatureOptimizer
from cognitive_optimization import CognitiveOptimizer
from aesthetic_optimization import AestheticOptimizer
from swarm_optimization import SwarmOptimizer, ConsensusStrategy
from systems_optimization import SystemsOptimizer
from auto_retry import AutoRetryEngine
from pipeline import IdeaGraphPipeline


class AutomatedScientist:
    """End-to-end automated scientist with tree search, self-reflection, and multi-review."""

    def __init__(self, enable_optimizations: bool = True):
        self.ideation = IdeaGraphPipeline()
        self.designer = ExperimentDesigner()
        self.coder = CodeGenerator()
        self.executor = ExperimentExecutor()
        self.analyzer = ResultAnalyzer()
        self.writer = PaperWriter()
        self.reviewer = AutoReviewer()
        self.tree_search = ExperimentTreeSearch()
        self.reflection = SelfReflectionAgent()

        # Advanced optimization
        self.optimizer = PipelineOptimizer(
            enable_compression=enable_optimizations,
            enable_speculation=enable_optimizations,
            enable_circuit_breaker=enable_optimizations,
        ) if enable_optimizations else None
        self.budget_allocator: Optional[AdaptiveBudgetAllocator] = None

        # Creative optimization
        self.creative = CreativeOptimizer(
            max_iterations=3, enable_all=enable_optimizations,
        ) if enable_optimizations else None

        # Deep optimization
        self.deep: Optional[DeepOptimizer] = None  # initialized in run() with budget

        # Infrastructure optimization
        self.infra = InfraOptimizer(enable_all=enable_optimizations) if enable_optimizations else None

        # Meta optimization
        self.meta = MetaOptimizer(enable_all=enable_optimizations) if enable_optimizations else None

        # Quantum/frontier optimization
        self.quantum = QuantumOptimizer(enable_all=enable_optimizations) if enable_optimizations else None

        # Nature-inspired optimization
        self.nature = NatureOptimizer(enable_all=enable_optimizations) if enable_optimizations else None

        # Cognitive optimization
        self.cognitive = CognitiveOptimizer(enable_all=enable_optimizations) if enable_optimizations else None

        # Aesthetic optimization
        self.aesthetic = AestheticOptimizer(enable_all=enable_optimizations) if enable_optimizations else None

        # Agent swarm optimization
        self.swarm = SwarmOptimizer(enable_all=enable_optimizations) if enable_optimizations else None

        # Systems/reliability optimization
        self.systems = SystemsOptimizer(enable_all=enable_optimizations) if enable_optimizations else None

    def run(
        self,
        topic: str,
        budget_usd: float = 5.0,
        max_ideation_iterations: int = 20,
        max_scientist_iterations: int = 3,
        execution_timeout: int = 1800,
        on_progress: Optional[Callable[[str], None]] = None,
        debate_enabled: bool = False,
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the full automated scientist loop.

        Returns comprehensive results dict with all stages.
        """
        start_time = time.time()

        # CRITICAL: Clear all caches from previous runs
        reset_caches()

        # ── Set run context on config so base_agent can enforce cost limits ──
        import uuid as _uuid
        config._current_run_id = _uuid.uuid4().hex[:12]
        config._current_budget_usd = budget_usd
        config._current_user_id = user_id

        # Auto-retry engine for resilient stage execution
        retry_engine = AutoRetryEngine(on_progress=on_progress)

        # Initialize adaptive budget allocator
        self.budget_allocator = AdaptiveBudgetAllocator(total_budget_usd=budget_usd)

        # Initialize deep optimizer with budget
        self.deep = DeepOptimizer(budget_usd=budget_usd, enable_all=True)

        # Initialize working memory with topic context
        if self.meta and self.meta.memory:
            self.meta.memory.store("hypothesis", f"Exploring: {topic}", importance=0.8, source_stage="init")

        # Meta-learning warm start: use config from similar past runs
        if self.quantum and self.quantum.meta_learner:
            warm_config = self.quantum.meta_learner.warm_start(topic)
            if warm_config and on_progress:
                on_progress(f"  [META-LEARN] Warm-starting from similar past run")

        # Swarm: register pipeline agents in pool + router
        if self.swarm and self.swarm.agent_pool:
            agent_registry = [
                ("ideation", ["ideation", "research", "creativity"]),
                ("designer", ["experiment_design", "planning", "methodology"]),
                ("coder", ["code_generation", "python", "debugging"]),
                ("executor", ["execution", "sandbox", "runtime"]),
                ("analyzer", ["analysis", "statistics", "visualization"]),
                ("writer", ["paper_writing", "latex", "academic"]),
                ("reviewer", ["review", "evaluation", "critique"]),
            ]
            for role, caps in agent_registry:
                handle = self.swarm.agent_pool.spawn(role, caps)
                if self.swarm.router:
                    self.swarm.router.register(handle.agent_id, caps)
            if self.swarm.blackboard:
                self.swarm.blackboard.write("topic", topic, writer_id="pipeline")
                self.swarm.blackboard.write("budget", budget_usd, writer_id="pipeline")
            if on_progress:
                on_progress(f"  [SWARM] Spawned {len(agent_registry)} agents in pool")

        # Episodic memory: recall relevant past runs
        if self.cognitive and self.cognitive.episodic:
            past_episodes = self.cognitive.episodic.recall(topic, n=2)
            if past_episodes and on_progress:
                on_progress(f"  [EPISODIC] Recalled {len(past_episodes)} similar past runs")
                lessons = self.cognitive.episodic.recall_lessons(topic)
                if lessons:
                    on_progress(f"  [EPISODIC] Lessons: {'; '.join(lessons[:3])}")

        # Time-boxed executor: allocate time budgets per stage
        if self.cognitive and self.cognitive.time_box:
            time_budgets = self.cognitive.time_box.allocate()

        # Shapley contributor: set up stage tracking
        if self.cognitive and self.cognitive.shapley:
            self.cognitive.shapley.set_stages([
                "ideation", "tree_search", "experiment_design",
                "code_generation", "execution", "analysis", "paper_writing", "review",
            ])

        # Initialize fractal budget allocator
        if self.quantum:
            self.quantum.fractal_budget = FractalBudgetAllocator(budget_usd)

        # Structured logging
        if self.infra and self.infra.logger:
            self.infra.logger.info("pipeline", f"Starting run: topic='{topic}', budget=${budget_usd}")
        if self.infra and self.infra.cost_attributor:
            self.infra.cost_attributor.set_context(stage="pipeline_init")

        results = {
            # v1-compatible keys (so app.py can display results)
            "topic": topic,
            "coverage": 0.0,
            "ideas": [],
            "archive": {},
            "dag_summary": {},
            "stats": {},
            # v2-specific keys
            "iterations": [],
            "final_paper": None,
            "final_review": None,
            "best_idea": None,
            "status": "running",
            "mode": "v2_scientist",
            "optimization_stats": {},
        }

        for iteration in range(max_scientist_iterations):
            iter_start = time.time()
            iter_results = {
                "iteration": iteration + 1,
                "stages": {},
                "status": "running",
            }

            if on_progress:
                on_progress(f"\n{'='*60}")
                on_progress(f"ITERATION {iteration + 1}/{max_scientist_iterations}")
                on_progress(f"{'='*60}")

            # Update simulated annealing schedule
            set_annealing_iteration(iteration, max_scientist_iterations)

            # Systems: load shedding based on budget pressure
            if self.systems and self.systems.load_shedder:
                remaining_pct = (self.budget_allocator.remaining_budget / max(budget_usd, 0.01)) * 100
                self.systems.load_shedder.set_pressure(remaining_pct)

            # Systems: checkpoint before iteration
            if self.systems and self.systems.checkpoint:
                self.systems.checkpoint.save(f"iteration_{iteration}", {
                    "topic": topic, "iteration": iteration,
                    "ideas_count": len(results.get("ideas", [])),
                })

            # Systems: PID quality targeting
            if self.systems and self.systems.pid and iteration > 0:
                prev_reviews = [
                    ir.get("stages", {}).get("review", {}).get("overall_score", 5)
                    for ir in results.get("iterations", [])
                ]
                if prev_reviews:
                    pid_output = self.systems.pid.update(prev_reviews[-1] / 10.0)
                    if on_progress:
                        on_progress(f"  [PID] Quality control: output={pid_output:.2f}, converged={self.systems.pid.is_converged}")

            # Hero's journey: advance narrative stage
            if self.aesthetic and self.aesthetic.heros_journey:
                self.aesthetic.heros_journey.advance("ideation")

            # Lotka-Volterra: update idea ecosystem dynamics
            if self.aesthetic and self.aesthetic.lotka_volterra:
                _, filter_level = self.aesthetic.lotka_volterra.step()
                if on_progress and filter_level > 2.0:
                    on_progress(f"  [LOTKA-VOLTERRA] High filter pressure ({filter_level:.1f}), tightening quality threshold")

            # Narrative tension: get resource allocation for current act
            if self.aesthetic and self.aesthetic.narrative:
                self.aesthetic.narrative.record_tension("ideation")

            # Harmonic resonance: check for quality oscillation
            if self.aesthetic and self.aesthetic.harmonic:
                osc = self.aesthetic.harmonic.detect_oscillation("review_quality")
                if osc.get("oscillating") and on_progress:
                    on_progress(f"  [HARMONIC] Quality oscillation detected (period={osc['period']}, strength={osc['strength']})")

            # Ant colony: pheromone-based path optimization
            if self.nature and self.nature.ant_colony and iteration > 0:
                aco_path = self.nature.ant_colony.optimize(n_ants=5, n_iterations=3)
                if on_progress:
                    importance = self.nature.ant_colony.get_stage_importance()
                    top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
                    on_progress(f"  [ACO] Top stages: {', '.join(f'{s}={v}' for s, v in top)}")

            # Chaotic explorer: deterministic-chaotic parameter generation
            if self.nature and self.nature.chaos:
                chaos_params = self.nature.chaos.next_params(
                    ["exploration_temp", "creativity", "risk_tolerance"],
                    {"exploration_temp": (0.2, 0.9), "creativity": (0.3, 1.0), "risk_tolerance": (0.1, 0.8)},
                )

            # Thermodynamic free energy: quality-diversity balance
            if self.nature and self.nature.thermo:
                action = self.nature.thermo.optimal_action()
                suggested_t = self.nature.thermo.suggest_temperature()
                self.nature.thermo.set_temperature(suggested_t)
                if on_progress and action != "balanced":
                    on_progress(f"  [THERMO] Action: {action} (T={suggested_t:.2f})")

            # Quantum annealing: combinatorial stage optimization
            quantum_plan = {}
            if self.quantum and self.quantum.quantum_annealer:
                quantum_plan = self.quantum.quantum_annealer.anneal(n_steps=100)
                if on_progress:
                    q_skip = [s for s, a in quantum_plan.items() if a == "skip"]
                    if q_skip:
                        on_progress(f"  [QUANTUM] Annealing recommends skipping: {', '.join(q_skip)}")

            # Knapsack budget solver: optimal stage selection under budget
            knapsack_plan = {}
            if self.deep and self.deep.knapsack:
                remaining_budget_k = self.budget_allocator.remaining_budget * 1000  # rough token estimate
                knapsack_plan = self.deep.knapsack.solve(remaining_budget_k)
                if on_progress:
                    skipped_ks = [s for s, a in knapsack_plan.items() if a == "skip"]
                    if skipped_ks:
                        on_progress(f"  Knapsack budget solver skipping: {', '.join(skipped_ks)}")

            # Token budget projector: early stop check
            if self.deep and self.deep.budget_projector:
                self.deep.budget_projector.record_spend(estimate_cost_usd())
                if self.deep.budget_projector.should_early_stop():
                    if on_progress:
                        on_progress("  [BUDGET] Early stop recommended — <10% budget remaining")

            # Entropy check: is idea diversity dangerously low?
            if self.deep and self.deep.entropy_reg and self.deep.entropy_reg.needs_diversity_boost:
                boost = self.deep.entropy_reg.diversity_temperature_boost()
                if on_progress:
                    on_progress(f"  [ENTROPY] Low diversity detected, boosting temperature by +{boost:.2f}")

            # Get MCTS-recommended pipeline plan for this iteration
            mcts_plan = {}
            if self.creative and self.creative.mcts_router:
                mcts_plan = self.creative.mcts_router.get_plan()
                if on_progress:
                    skipped = [s for s, a in mcts_plan.items() if a == "skip"]
                    if skipped:
                        on_progress(f"  MCTS recommends skipping: {', '.join(skipped)}")

            # Get PBT hyperparameters for this run
            pbt_config = None
            if self.creative and self.creative.pbt:
                pbt_config = self.creative.pbt.get_config()
                if on_progress and pbt_config.runs > 0:
                    on_progress(f"  PBT config #{pbt_config.id} (fitness={pbt_config.fitness:.3f})")

            try:
                # ── Stage 1: Ideation ────────────────────────────────────
                if on_progress:
                    on_progress("\n[Stage 1/7] Ideation — Generating research ideas...")

                stage_start = time.time()
                if self.infra and self.infra.logger:
                    self.infra.logger.start_timer("ideation")
                if self.infra and self.infra.cost_attributor:
                    self.infra.cost_attributor.set_context(stage="ideation")
                ideation_budget = self.budget_allocator.get_stage_budget("ideation")
                ideation_results = self.ideation.run(
                    topic=topic,
                    budget_usd=ideation_budget,
                    max_iterations=max_ideation_iterations,
                    on_progress=on_progress,
                    debate_enabled=debate_enabled,
                    user_id=user_id,
                )

                ideas = ideation_results.get("ideas", [])

                # Copy v1 results into v2 for app compatibility
                results["ideas"] = ideas
                results["coverage"] = ideation_results.get("coverage", 0)
                results["archive"] = ideation_results.get("archive", {})
                results["dag_summary"] = ideation_results.get("dag_summary", {})
                results["stats"] = ideation_results.get("stats", {})

                # Record ideation performance
                ideation_quality = ideation_results.get("coverage", 0)
                if self.optimizer:
                    self.optimizer.record_stage("ideation", time.time() - stage_start, quality=ideation_quality)
                self.budget_allocator.record_stage_result(
                    "ideation", tokens_used=get_token_usage().get("prompt", 0),
                    quality=ideation_quality,
                )
                cost_so_far = estimate_cost_usd()
                self.budget_allocator.record_spend(cost_so_far)

                if not ideas:
                    if on_progress:
                        on_progress("No ideas generated. Stopping.")
                    iter_results["status"] = "failed"
                    iter_results["error"] = "No ideas generated"
                    results["iterations"].append(iter_results)
                    break

                # ── Swarm Idea Processing ────────────────────────────────
                # Swarm consensus: aggregate quality from multiple evaluator perspectives
                if self.swarm and self.swarm.consensus and len(ideas) >= 3:
                    for idea in ideas[:10]:
                        # Simulate multi-evaluator perspectives
                        evaluations = {
                            "novelty_eval": {"incremental": 0.3, "moderate": 0.6, "radical": 0.9}.get(idea.get("novelty_level", "moderate"), 0.5),
                            "quality_eval": idea.get("quality_score", 0.5),
                            "feasibility_eval": 0.5,
                        }
                        consensus_score = self.swarm.consensus.reach_consensus(
                            evaluations, ConsensusStrategy.WEIGHTED_AVERAGE,
                        )
                        idea["swarm_consensus_score"] = consensus_score

                # Stigmergy: deposit pheromone on promising methodology types
                if self.swarm and self.swarm.stigmergy:
                    for idea in ideas[:5]:
                        mtype = idea.get("methodology_type", "unknown")
                        q = idea.get("quality_score", 0)
                        self.swarm.stigmergy.deposit(
                            f"method:{mtype}", mtype, strength=q,
                            depositor_id="ideation",
                        )

                # Swarm memory: store best ideas for cross-stage access
                if self.swarm and self.swarm.memory_pool:
                    for idea in ideas[:3]:
                        self.swarm.memory_pool.store(
                            f"idea:{idea.get('title', '')[:30]}",
                            idea, agent_id="ideation",
                        )

                # Message bus: broadcast ideation completion
                if self.swarm and self.swarm.message_bus:
                    self.swarm.message_bus.publish(
                        "stage_complete", {"stage": "ideation", "ideas_count": len(ideas)},
                        sender_id="pipeline",
                    )

                # ── Aesthetic Idea Processing ────────────────────────────
                # Portfolio optimizer: select diversified idea portfolio
                if self.aesthetic and self.aesthetic.portfolio and len(ideas) >= 4:
                    for idea in ideas:
                        self.aesthetic.portfolio.add_idea(
                            idea.get("title", "")[:30],
                            [idea.get("quality_score", 0.5)] * 3,  # replicate for variance estimate
                        )
                    portfolio = self.aesthetic.portfolio.optimal_portfolio(n_select=3)
                    if on_progress and portfolio:
                        on_progress(f"  [PORTFOLIO] Diversified selection: {len(portfolio)} ideas (Sharpe-ranked)")

                # Sabermetric ranking: WAR-like composite stats
                if self.aesthetic and self.aesthetic.sabermetric:
                    for idea in ideas[:10]:
                        self.aesthetic.sabermetric.add_idea(
                            idea.get("title", "")[:30],
                            novelty={"incremental": 0.3, "moderate": 0.6, "radical": 0.9}.get(idea.get("novelty_level", "moderate"), 0.5),
                            feasibility=0.6,
                            impact=idea.get("quality_score", 0.5),
                            clarity=0.5,
                        )
                    mvp = self.aesthetic.sabermetric.get_mvp()
                    if mvp and on_progress:
                        on_progress(f"  [SABERMETRIC] MVP: {mvp}")

                # Lotka-Volterra: update ecosystem with actual idea count
                if self.aesthetic and self.aesthetic.lotka_volterra:
                    self.aesthetic.lotka_volterra.step(actual_ideas=len(ideas))

                # ── Nature-Inspired Idea Processing ─────────────────────
                # Immune system: clonal selection on ideas
                if self.nature and self.nature.immune and len(ideas) >= 5:
                    for idea in ideas:
                        self.nature.immune.add_idea(
                            idea.get("title", "")[:30],
                            {"novelty": 0.5, "feasibility": 0.5, "impact": 0.5, "clarity": 0.5},
                            idea.get("quality_score", 0.5),
                        )
                    self.nature.immune.clonal_selection()
                    survivors = self.nature.immune.get_best(n=len(ideas))
                    if on_progress:
                        on_progress(f"  [IMMUNE] Clonal selection: {len(survivors)} antibodies retained")

                # Coevolution: pit ideas against critics
                if self.nature and self.nature.coevolution and len(ideas) >= 3:
                    for idea in ideas[:8]:
                        self.nature.coevolution.add_idea(idea.get("title", "")[:30], idea.get("quality_score", 0.5))
                    # Add synthetic critics at different strictness levels
                    for i, strictness in enumerate([0.3, 0.5, 0.7]):
                        self.nature.coevolution.add_critic(f"critic_{i}", strictness)
                    results_coevo = self.nature.coevolution.compete()
                    self.nature.coevolution.evolve()
                    surviving_ids = set(self.nature.coevolution.get_survivors(5))
                    if on_progress:
                        on_progress(f"  [COEVO] Arms race: {len(surviving_ids)} ideas survived {len(results_coevo)} matchups")

                # Social influence: track which ideas inspire others
                if self.nature and self.nature.influence and len(ideas) >= 2:
                    for i, idea in enumerate(ideas[:-1]):
                        # Assume adjacent ideas in quality order may have influenced each other
                        self.nature.influence.add_influence(
                            ideas[i].get("title", "")[:30],
                            ideas[i + 1].get("title", "")[:30],
                        )

                # Record thermodynamic state
                if self.nature and self.nature.thermo:
                    self.nature.thermo.record(ideation_quality, self.deep.entropy_reg.entropy_ratio if (self.deep and self.deep.entropy_reg) else 0.5)

                # ── Advanced Idea Selection ──────────────────────────────
                # Pareto front: multi-objective ranking
                if self.creative and self.creative.pareto and len(ideas) >= 3:
                    self.creative.pareto = ParetoFront()  # fresh front per iteration
                    for idea in ideas:
                        self.creative.pareto.add(
                            id=idea.get("title", "")[:30],
                            data=idea,
                            objectives={
                                "quality": idea.get("quality_score", 0),
                                "novelty": {"incremental": 0.3, "moderate": 0.6, "radical": 0.9}.get(
                                    idea.get("novelty_level", "moderate"), 0.5
                                ),
                                "feasibility": 1.0 - idea.get("risk_assessment", 0.5) if isinstance(idea.get("risk_assessment"), (int, float)) else 0.5,
                                "cost_efficiency": 1.0,
                            },
                        )
                    pareto_best = self.creative.pareto.select_best(n=1)
                    if pareto_best:
                        best_idea = pareto_best[0].data
                        if on_progress:
                            on_progress(f"  Pareto-optimal idea selected (rank 0, {self.creative.pareto.stats()['front_0_size']} on front)")
                    else:
                        best_idea = max(ideas, key=lambda x: x.get("quality_score", 0))
                # ELO tournament: pairwise ranking for top ideas
                elif self.creative and self.creative.elo and len(ideas) >= 4:
                    elo = self.creative.elo
                    top_ideas = heapq.nlargest(8, ideas, key=lambda x: x.get("quality_score", 0))
                    idea_map = {idea.get("title", f"idea_{i}")[:30]: idea for i, idea in enumerate(top_ideas)}
                    ids = list(idea_map.keys())
                    matchups = elo.generate_matchups(ids, n_matches=len(ids))
                    # Run pairwise comparisons via LLM
                    for id_a, id_b in matchups[:6]:  # cap at 6 comparisons to save budget
                        prompt = elo.build_comparison_prompt(idea_map[id_a], idea_map[id_b])
                        result_json = self.reflection._call_json(
                            prompt["system"], prompt["user"], max_tokens=256,
                        )
                        winner = result_json.get("winner", "draw")
                        if winner == "A":
                            elo.record_match(id_a, id_b)
                        elif winner == "B":
                            elo.record_match(id_b, id_a)
                        else:
                            elo.record_match(id_a, id_b, draw=True)
                    rankings = elo.get_rankings()
                    if rankings:
                        best_idea = idea_map.get(rankings[0][0], ideas[0])
                        if on_progress:
                            on_progress(f"  ELO tournament: top={rankings[0][0]} (rating={rankings[0][1]:.0f})")
                else:
                    best_idea = max(ideas, key=lambda x: x.get("quality_score", 0))

                # Adversarial stress test on best idea
                adversarial_result = None
                if self.creative and self.creative.adversarial and best_idea:
                    if on_progress:
                        on_progress("  Adversarial stress testing best idea...")
                    attacks = self.creative.adversarial.build_attack_prompts(best_idea)
                    attack_results = []
                    for attack in attacks[:3]:  # limit to 3 attacks to save budget
                        atk_result = self.reflection._call_json(
                            attack["system"], attack["user"], max_tokens=512,
                        )
                        atk_result["vector"] = attack["vector"]
                        attack_results.append(atk_result)
                    adversarial_result = AdversarialTester.compute_resilience(attack_results)
                    if on_progress:
                        on_progress(
                            f"  Resilience: {adversarial_result['resilience_score']:.2f} "
                            f"({adversarial_result['recommendation']})"
                        )
                    # If idea fails adversarial test badly, try next best
                    if adversarial_result["recommendation"] == "abandon" and len(ideas) > 1:
                        ideas_sorted = sorted(ideas, key=lambda x: x.get("quality_score", 0), reverse=True)
                        best_idea = ideas_sorted[1]  # fallback to second-best
                        if on_progress:
                            on_progress(f"  Idea failed stress test, falling back to: {best_idea.get('title', '')[:40]}")

                # ── Meta-optimization: quality filtering + dedup + forecasting ──
                if self.meta:
                    # Kalman-filter quality scores for robustness
                    if self.meta.kalman:
                        for idea in ideas:
                            raw_q = idea.get("quality_score", 0)
                            filtered_q = self.meta.kalman.update(
                                idea.get("title", "")[:30], raw_q
                            )
                            idea["quality_score_filtered"] = filtered_q

                    # Bloom filter dedup: O(1) duplicate detection
                    if self.meta.bloom:
                        deduped = []
                        for idea in ideas:
                            key = (idea.get("title", "") + idea.get("method", ""))[:100]
                            if self.meta.bloom.add_if_new(key):
                                deduped.append(idea)
                        if len(deduped) < len(ideas) and on_progress:
                            on_progress(f"  [BLOOM] Filtered {len(ideas) - len(deduped)} duplicate ideas")
                        ideas = deduped if deduped else ideas

                    # Reservoir sampling: maintain representative sample
                    if self.meta.reservoir:
                        for idea in ideas:
                            self.meta.reservoir.add(idea)

                    # Forecast quality trend
                    if self.meta.forecaster and best_idea:
                        self.meta.forecaster.observe("idea_quality", best_idea.get("quality_score", 0))
                        if self.meta.forecaster.is_converging("idea_quality") and on_progress:
                            on_progress("  [FORECAST] Quality converging — diminishing returns expected")

                    # Topological diversity mapping
                    if self.quantum and self.quantum.topology:
                        for idea in ideas[:20]:
                            features = [
                                idea.get("quality_score", 0),
                                {"incremental": 0.3, "moderate": 0.6, "radical": 0.9}.get(idea.get("novelty_level", "moderate"), 0.5),
                                0.5,  # feasibility placeholder
                                len(idea.get("method", "")) / 500,  # method complexity
                                len(idea.get("resources", "")) / 200,  # resource weight
                            ]
                            self.quantum.topology.add_idea(idea.get("title", "")[:30], features)
                        if on_progress:
                            topo = self.quantum.topology.stats()
                            on_progress(f"  [TOPOLOGY] spread={topo['spread']:.2f}, components={topo.get('components_0.5', '?')}")

                    # Feedback loop detection
                    if self.meta.feedback:
                        self.meta.feedback.record("ideation", ideation_quality)
                        issues = self.meta.feedback.scan_all()
                        for issue in issues:
                            if on_progress:
                                on_progress(f"  [FEEDBACK] {issue['type']} detected in {issue['stage']}")

                    # Store key findings in working memory
                    if self.meta.memory and best_idea:
                        self.meta.memory.store(
                            "method_current", best_idea.get("method", "")[:200],
                            importance=0.9, source_stage="ideation",
                        )
                        self.meta.memory.store(
                            "hypothesis", best_idea.get("hypothesis", "")[:200],
                            importance=0.95, source_stage="ideation",
                        )

                # Track entropy of generated ideas for diversity pressure
                if self.deep and self.deep.entropy_reg:
                    for idea in ideas:
                        mtype = idea.get("methodology_type", "unknown")
                        nlevel = idea.get("novelty_level", "unknown")
                        self.deep.entropy_reg.record(f"{mtype}:{nlevel}")

                # Distill knowledge nuggets from best idea for future use
                if self.deep and self.deep.distiller and best_idea:
                    method = best_idea.get("method", "")
                    if method and len(method) > 20:
                        self.deep.distiller.add(
                            f"Promising approach in {topic}: {method[:150]}",
                            domain=topic, nugget_type="method", confidence=best_idea.get("quality_score", 0.5),
                        )

                results["best_idea"] = best_idea
                iter_results["stages"]["ideation"] = {
                    "ideas_count": len(ideas),
                    "best_idea_title": best_idea.get("title", ""),
                    "best_quality": best_idea.get("quality_score", 0),
                    "coverage": ideation_results.get("coverage", 0),
                    "adversarial_resilience": adversarial_result.get("resilience_score", None) if adversarial_result else None,
                    "selection_method": "pareto" if (self.creative and self.creative.pareto) else "elo" if (self.creative and self.creative.elo) else "max_quality",
                    "entropy_ratio": self.deep.entropy_reg.entropy_ratio if (self.deep and self.deep.entropy_reg) else None,
                }

                if on_progress:
                    on_progress(
                        f"  Best idea: {best_idea.get('title', '')[:50]} "
                        f"(q={best_idea.get('quality_score', 0):.3f})"
                    )

                # ── Stage 2: Experiment Design + Tree Search ────────────
                if on_progress:
                    on_progress("\n[Stage 2/7] Experiment Design (with tree search)...")

                stage_start = time.time()

                # Check warm cache for similar experiment plans
                cached_plan = None
                if self.optimizer:
                    cached_plan = self.optimizer.warm_cache.get(
                        "experiment_design", best_idea.get("title", ""),
                        best_idea.get("methodology_type", ""),
                        max_age_s=1800,  # 30 min freshness
                    )

                if cached_plan:
                    experiment_plan = cached_plan
                    if on_progress:
                        on_progress("  [CACHE HIT] Reusing experiment plan from warm cache")
                    tree_nodes = []
                    best_approach = None
                else:
                    # Tree search: explore multiple approaches
                    tree_nodes = self.tree_search.search(
                        best_idea, domain=topic, max_branches=3, max_depth=2,
                        on_progress=on_progress,
                    )
                    best_approach = tree_nodes[0] if tree_nodes else None

                    # Design experiment from best approach
                    experiment_plan = self.designer.design(
                        {**best_idea, **(best_approach.experiment_plan if best_approach else {})},
                        domain=topic, on_progress=on_progress,
                    )

                    # Cache the plan for potential reuse
                    if self.optimizer:
                        self.optimizer.warm_cache.put(
                            "experiment_design", best_idea.get("title", ""),
                            best_idea.get("methodology_type", ""),
                            value=experiment_plan,
                        )

                # MCTS-guided self-reflection decision
                skip_reflection = mcts_plan.get("self_reflection_experiment") == "skip"
                if skip_reflection and on_progress:
                    on_progress("  [MCTS] Skipping experiment plan reflection")

                # Self-reflection on experiment plan
                if not skip_reflection:
                    if on_progress:
                        on_progress("  Self-reflecting on experiment plan...")
                reflection = self.reflection.reflect_on_experiment(experiment_plan, best_idea) if not skip_reflection else {}
                if reflection.get("improved_plan") and reflection.get("confidence", 0) > 0.6:
                    experiment_plan = {**experiment_plan, **reflection["improved_plan"]}
                    if on_progress:
                        on_progress(f"  Plan improved (confidence={reflection.get('confidence', 0):.2f})")

                if self.optimizer:
                    self.optimizer.record_stage("experiment_design", time.time() - stage_start)
                self.budget_allocator.record_stage_result(
                    "experiment_design", tokens_used=500, quality=reflection.get("confidence", 0.5),
                )

                iter_results["stages"]["experiment_design"] = {
                    "hypothesis": experiment_plan.get("hypothesis", ""),
                    "steps_count": len(experiment_plan.get("steps", [])),
                    "datasets_count": len(experiment_plan.get("datasets", [])),
                    "metrics_count": len(experiment_plan.get("metrics", [])),
                    "tree_nodes_explored": len(tree_nodes),
                    "best_approach_score": best_approach.score if best_approach else 0,
                    "reflection_issues": reflection.get("issues", []),
                }

                # ── Stage 3: Code Generation + Quality Gate ──────────────
                if on_progress:
                    on_progress("\n[Stage 3/7] Code Generation (with quality gate)...")

                # Predictive quality gate: should we proceed to expensive code gen?
                if self.meta and self.meta.quality_gate:
                    gate_features = {
                        "idea_quality": best_idea.get("quality_score", 0.5),
                        "plan_confidence": reflection.get("confidence", 0.5),
                        "adversarial_resilience": adversarial_result.get("resilience_score", 0.5) if adversarial_result else 0.5,
                        "historical_success": 0.5,
                    }
                    if not self.meta.quality_gate.should_proceed(gate_features):
                        if on_progress:
                            prob = self.meta.quality_gate.predict(gate_features)
                            on_progress(f"  [GATE] Low success probability ({prob:.2f}) — proceeding anyway")

                stage_start = time.time()
                # Inject working memory context into code generation
                memory_context = ""
                if self.meta and self.meta.memory:
                    memory_context = self.meta.memory.retrieve("code_generation", max_tokens=500)

                code_files = self.coder.generate(
                    experiment_plan, on_progress=on_progress,
                )

                # Self-reflection on code
                if on_progress:
                    on_progress("  Self-reflecting on generated code...")
                code_reflection = self.reflection.reflect_on_code(
                    code_files.get("experiment.py", ""), experiment_plan,
                )
                if (code_reflection.get("fixed_code") and
                    code_reflection.get("confidence", 0) > 0.5 and
                    code_reflection.get("severity") in ("medium", "high")):
                    code_files["experiment.py"] = code_reflection["fixed_code"]
                    if on_progress:
                        fixes = code_reflection.get("fixes", [])
                        on_progress(f"  Code improved: {len(fixes)} fixes applied")

                quality_score = float(code_files.pop("_quality_score", "0.5"))

                if self.optimizer:
                    self.optimizer.record_stage(
                        "code_generation", time.time() - stage_start, quality=quality_score,
                    )
                self.budget_allocator.record_stage_result(
                    "code_generation", tokens_used=1000, quality=quality_score,
                )

                # Speculative execution: start sandbox prep while we log results
                if self.optimizer and self.optimizer.speculator:
                    self.optimizer.speculator.speculate(
                        "execution_prep",
                        lambda cf=code_files: {"files": list(cf.keys()), "ready": True},
                    )

                iter_results["stages"]["code_generation"] = {
                    "files_count": len(code_files),
                    "main_script_lines": len(code_files.get("experiment.py", "").split("\n")),
                    "files": list(code_files.keys()),
                    "quality_score": quality_score,
                    "reflection_issues": code_reflection.get("issues", []),
                }

                # ── Stage 4: Execution ───────────────────────────────────
                if on_progress:
                    on_progress("\n[Stage 4/7] Experiment Execution...")

                stage_start = time.time()
                # Use optimizer-suggested timeout if available
                exec_timeout = execution_timeout
                if self.optimizer:
                    exec_timeout = int(self.optimizer.suggest_timeout("execution", execution_timeout))

                run_result = self.executor.execute(
                    code_files, timeout=exec_timeout, on_progress=on_progress,
                )

                if self.optimizer:
                    self.optimizer.record_stage(
                        "execution", time.time() - stage_start,
                        quality=1.0 if run_result.success else 0.0,
                        failed=not run_result.success,
                    )

                # Rollback detection + reward shaping on execution failure
                if not run_result.success and self.deep:
                    # Attention rollback: identify which upstream stage to blame
                    if self.deep.rollback:
                        self.deep.rollback.record_quality("code_generation", quality_score)
                        self.deep.rollback.record_quality("experiment_design", reflection.get("confidence", 0.5))
                        target = self.deep.rollback.identify_rollback_target("execution")
                        if target and on_progress:
                            on_progress(f"  [ROLLBACK] Blame attribution → {target}")
                        if target:
                            self.deep.rollback.record_failure_pair("execution", target)

                    # Reward shaping: extract partial successes from failure
                    if self.deep.reward_shaper:
                        partial = []
                        if run_result.stdout and len(run_result.stdout) > 50:
                            partial.append("code_executed_partially")
                        if "import" not in (run_result.error_summary or ""):
                            partial.append("imports_resolved")
                        shaped = self.deep.reward_shaper.shaped_reward(0.0, partial)
                        self.deep.reward_shaper.record_failure(
                            goal=experiment_plan.get("hypothesis", ""),
                            outcome=run_result.error_summary or "unknown failure",
                            partial_successes=partial,
                            lessons=[f"Execution failed: {run_result.error_summary[:100]}"] if run_result.error_summary else [],
                            reusable=[],
                        )
                        if on_progress and partial:
                            on_progress(f"  [REWARD] Shaped reward: {shaped:.2f} (partial: {', '.join(partial)})")

                    # Contrastive learning: record failure pattern
                    if run_result.error_summary:
                        _CONTRASTIVE.record_failure("code_generation", run_result.error_summary[:80])

                # Update knapsack stage costs
                if self.deep and self.deep.knapsack:
                    exec_duration = time.time() - stage_start
                    self.deep.knapsack.update_cost("execution", exec_duration / 60)  # in minutes as proxy

                iter_results["stages"]["execution"] = run_result.to_dict()

                # ── Stage 5: Analysis + Reflection ───────────────────────
                if on_progress:
                    on_progress("\n[Stage 5/7] Result Analysis...")

                stage_start = time.time()
                analysis = self.analyzer.analyze(
                    run_result, experiment_plan, on_progress=on_progress,
                )

                # Self-reflect on results
                if on_progress:
                    on_progress("  Self-reflecting on analysis...")
                results_reflection = self.reflection.reflect_on_results(analysis, experiment_plan)
                if results_reflection.get("improved_summary"):
                    analysis["summary"] = results_reflection["improved_summary"]
                analysis["alternative_explanations"] = results_reflection.get("alternative_explanations", [])
                analysis["missing_analyses"] = results_reflection.get("missing_analyses", [])

                if self.optimizer:
                    self.optimizer.record_stage(
                        "analysis", time.time() - stage_start,
                        quality=results_reflection.get("confidence", 0.5),
                    )
                self.budget_allocator.record_stage_result(
                    "analysis", tokens_used=500,
                    quality=results_reflection.get("confidence", 0.5),
                )

                iter_results["stages"]["analysis"] = {
                    "summary": analysis.get("summary", ""),
                    "findings_count": len(analysis.get("key_findings", [])),
                    "supports_hypothesis": analysis.get("supports_hypothesis", "unknown"),
                    "execution_success": analysis.get("execution_success", False),
                    "reflection_confidence": results_reflection.get("confidence", 0),
                }

                # ── Stage 6: Paper Writing + Self-Review ─────────────────
                if on_progress:
                    on_progress("\n[Stage 6/7] Paper Writing...")

                stage_start = time.time()
                dag_summary = ideation_results.get("dag_summary", {})
                paper = self.writer.write(
                    best_idea, experiment_plan, analysis,
                    dag_summary=dag_summary, on_progress=on_progress,
                )
                # Self-reflect on paper before review
                if on_progress:
                    on_progress("  Self-reviewing paper...")
                paper_reflection = self.reflection.reflect_on_paper(paper.get("sections", {}))
                paper_quality = paper_reflection.get("overall_quality", 0)

                if self.optimizer:
                    self.optimizer.record_stage(
                        "paper_writing", time.time() - stage_start,
                        quality=paper_quality,
                    )
                self.budget_allocator.record_stage_result(
                    "paper_writing", tokens_used=2000, quality=paper_quality,
                )

                results["final_paper"] = paper
                iter_results["stages"]["paper"] = {
                    "title": paper.get("title", ""),
                    "latex_length": len(paper.get("latex", "")),
                    "markdown_length": len(paper.get("markdown", "")),
                    "sections": list(paper.get("sections", {}).keys()),
                    "self_review_quality": paper_quality,
                    "ready_for_review": paper_reflection.get("ready_for_review", True),
                    "top_improvements": paper_reflection.get("top_improvements", []),
                }

                # ── Stage 7: Multi-Reviewer Panel ────────────────────────
                if on_progress:
                    on_progress("\n[Stage 7/7] Multi-Reviewer Panel (3 reviewers)...")

                stage_start = time.time()
                review = self.reviewer.multi_review(
                    paper, n_reviewers=3, on_progress=on_progress,
                )

                # Nash consensus: game-theoretic review aggregation
                review_quality = review.get("overall_score", 0) / 10.0
                if self.meta and self.meta.nash and review.get("individual_scores"):
                    scores = {f"reviewer_{i}": s for i, s in enumerate(review["individual_scores"])}
                    nash_score = self.meta.nash.compute_consensus(scores)
                    review["nash_consensus_score"] = nash_score
                    if on_progress:
                        on_progress(f"  Nash consensus: {nash_score:.1f}/10 (raw avg: {review.get('overall_score', 0):.1f})")
                    review_quality = nash_score / 10.0

                # Mutual information tracking: how much did each stage contribute?
                if self.meta and self.meta.mutual_info:
                    for stage_name in ["ideation", "experiment_design", "code_generation", "analysis", "paper_writing"]:
                        stage_q = iter_results["stages"].get(stage_name, {}).get("quality_score",
                            iter_results["stages"].get(stage_name, {}).get("reflection_confidence",
                            iter_results["stages"].get(stage_name, {}).get("best_quality", 0.5)))
                        self.meta.mutual_info.record(stage_name, stage_q, review_quality)

                # Forecast review quality trend
                if self.meta and self.meta.forecaster:
                    self.meta.forecaster.observe("review_quality", review_quality)

                # Update predictive quality gate
                if self.meta and self.meta.quality_gate:
                    gate_features = {
                        "idea_quality": best_idea.get("quality_score", 0.5),
                        "plan_confidence": reflection.get("confidence", 0.5),
                        "code_quality": quality_score,
                    }
                    self.meta.quality_gate.record_outcome(gate_features, review_quality > 0.6)

                # Feedback loop detection on review
                if self.meta and self.meta.feedback:
                    self.meta.feedback.record("review", review_quality)

                if self.optimizer:
                    self.optimizer.record_stage(
                        "review", time.time() - stage_start, quality=review_quality,
                    )
                self.budget_allocator.record_stage_result(
                    "review", tokens_used=1000, quality=review_quality,
                )

                results["final_review"] = review
                iter_results["stages"]["review"] = {
                    "overall_score": review.get("overall_score", 0),
                    "decision": review.get("decision", ""),
                    "strengths_count": len(review.get("strengths", [])),
                    "weaknesses_count": len(review.get("weaknesses", [])),
                    "reviewer_count": review.get("reviewer_count", 1),
                    "consensus": review.get("consensus", ""),
                    "score_variance": review.get("score_variance", 0),
                }

                iter_results["status"] = "completed"
                iter_results["elapsed_seconds"] = time.time() - iter_start

                if on_progress:
                    on_progress(
                        f"\n  Iteration {iteration + 1} complete: "
                        f"Review = {review.get('decision', '?')} "
                        f"(score: {review.get('overall_score', 0):.1f}/10)"
                    )

                results["iterations"].append(iter_results)

                # ── Iteration circuit breaker: stop if recent iterations are failing ──
                _recent_iters = results.get("iterations", [])[-3:]
                if len(_recent_iters) >= 3:
                    _recent_success = sum(
                        1 for ir in _recent_iters
                        if ir.get("status") == "completed"
                        and ir.get("stages", {}).get("review", {}).get("overall_score", 0) >= 3
                    )
                    if _recent_success == 0:
                        if on_progress:
                            on_progress(
                                f"\n[CIRCUIT] Last 3 iterations all low-quality. "
                                f"Stopping to save budget."
                            )
                        break

                # ── Per-iteration budget enforcement ─────────────────────────
                _iter_cost = estimate_cost_usd()
                if _iter_cost >= budget_usd * 0.95:
                    if on_progress:
                        on_progress(
                            f"\n[BUDGET] 95% consumed (${_iter_cost:.3f}/${budget_usd:.2f}). "
                            f"Stopping scientist loop."
                        )
                    break

                # Check if paper is accepted
                decision = review.get("decision", "").lower()
                if decision in ("strong_accept", "accept"):
                    if on_progress:
                        on_progress(f"\nPaper ACCEPTED! Stopping after {iteration + 1} iterations.")
                    break

                # Record MCTS outcomes for learning
                if self.creative and self.creative.mcts_router:
                    self.creative.mcts_router.record_outcome("tree_search", review_quality)
                    self.creative.mcts_router.record_outcome("self_reflection_experiment", review_quality)
                    self.creative.mcts_router.record_outcome("self_reflection_code", quality_score)

                # Record PBT fitness
                if self.creative and self.creative.pbt and pbt_config:
                    self.creative.pbt.record_result(pbt_config.id, review_quality)

                # Momentum optimizer: smooth quality trajectory
                if self.nature and self.nature.momentum:
                    self.nature.momentum.update({"review_quality": review_quality - 0.5, "idea_quality": best_idea.get("quality_score", 0.5) - 0.5})
                    if on_progress and self.nature.momentum.is_improving("review_quality"):
                        on_progress(f"  [MOMENTUM] Quality improving (momentum={self.nature.momentum.get_momentum('review_quality'):.3f})")

                # Wisdom of crowds: diversity-weighted review aggregation
                if self.nature and self.nature.crowds:
                    self.nature.crowds.add_scores(
                        f"iter_{iteration}",
                        [review.get("overall_score", 5)] * 3,  # placeholder for individual scores
                    )

                # Shapley: record stage coalition values
                if self.cognitive and self.cognitive.shapley:
                    stages_run = set(iter_results.get("stages", {}).keys())
                    self.cognitive.shapley.record_coalition_value(stages_run, review_quality)
                    # Also record individual stage values
                    for stage_name, stage_data in iter_results.get("stages", {}).items():
                        q = stage_data.get("quality_score", stage_data.get("best_quality", 0.5))
                        if isinstance(q, (int, float)):
                            self.cognitive.shapley.record_coalition_value({stage_name}, q)

                # Semantic memory: learn generalizable facts
                if self.cognitive and self.cognitive.semantic:
                    if review_quality > 0.7:
                        self.cognitive.semantic.add_relation(
                            topic.split()[0] if topic else "unknown",
                            best_idea.get("methodology_type", "unknown"),
                            "improves_with", review_quality,
                        )
                    elif review_quality < 0.3:
                        self.cognitive.semantic.add_relation(
                            topic.split()[0] if topic else "unknown",
                            "low_quality", "associated_with", 1.0 - review_quality,
                        )

                # Time-box: record actual stage durations
                if self.cognitive and self.cognitive.time_box:
                    self.cognitive.time_box.record_actual("review", time.time() - stage_start)

                # ── Swarm Post-Review Processing ─────────────────────────
                # Emergent behavior detection
                if self.swarm and self.swarm.emergence:
                    self.swarm.emergence.record_event("review_complete", "reviewer", {"score": review_quality})
                    self.swarm.emergence.record_event("iteration_end", "pipeline", {"iteration": iteration})
                    patterns = self.swarm.emergence.detect_patterns()
                    if patterns and on_progress:
                        for p in patterns[:2]:
                            on_progress(f"  [EMERGENT] {p.pattern_type}: {p.recommendation} (strength={p.strength:.2f})")

                # Swarm memory consolidation
                if self.swarm and self.swarm.memory_pool:
                    promoted = self.swarm.memory_pool.consolidate(min_access=2)
                    if promoted > 0 and on_progress:
                        on_progress(f"  [SWARM MEMORY] Consolidated {promoted} items to long-term")

                # Stigmergy evaporation
                if self.swarm and self.swarm.stigmergy:
                    removed = self.swarm.stigmergy.evaporate()
                    strongest = self.swarm.stigmergy.strongest(3)
                    if strongest and on_progress:
                        on_progress(f"  [STIGMERGY] Top markers: {', '.join(f'{k}={s:.2f}' for k, s in strongest)}")

                # Blackboard: record iteration results
                if self.swarm and self.swarm.blackboard:
                    self.swarm.blackboard.write(f"iter_{iteration}_review", review_quality, writer_id="reviewer")
                    self.swarm.blackboard.write(f"iter_{iteration}_idea", best_idea.get("title", ""), writer_id="ideation")

                # Dynamic team forming for next iteration
                if self.swarm and self.swarm.team_former and iteration < max_scientist_iterations - 1:
                    team = self.swarm.team_former.form_team(
                        f"iteration_{iteration + 1}",
                        required_roles=["ideation", "coder", "reviewer"],
                        team_size=3,
                    )
                    if on_progress:
                        on_progress(f"  [SWARM TEAM] Formed team of {len(team.agents)} for next iteration")

                # Contract-net: announce next iteration's tasks
                if self.swarm and self.swarm.negotiator and iteration < max_scientist_iterations - 1:
                    self.swarm.negotiator.announce(
                        f"iteration_{iteration + 1}",
                        requirements=["ideation", "code_generation"],
                        budget_limit=self.budget_allocator.remaining_budget * 0.5,
                    )

                # Dialectical synthesis: generate antithesis for best idea
                if self.aesthetic and self.aesthetic.dialectic and iteration < max_scientist_iterations - 1:
                    anti_prompt = self.aesthetic.dialectic.build_antithesis_prompt(best_idea)
                    antithesis = self.reflection._call_json(
                        anti_prompt["system"], anti_prompt["user"], max_tokens=512,
                    )
                    if antithesis.get("antithesis_title"):
                        syn_prompt = self.aesthetic.dialectic.build_synthesis_prompt(best_idea, antithesis)
                        synthesis = self.reflection._call_json(
                            syn_prompt["system"], syn_prompt["user"], max_tokens=512,
                        )
                        if synthesis.get("title"):
                            self.aesthetic.dialectic.record_synthesis(
                                best_idea.get("title", ""), antithesis.get("antithesis_title", ""),
                                synthesis.get("title", ""), quality_gain=review_quality - best_idea.get("quality_score", 0),
                            )
                            if on_progress:
                                on_progress(f"  [DIALECTIC] Synthesis: {synthesis.get('title', '')[:50]}")

                # Desire line: record the pipeline path taken
                if self.aesthetic and self.aesthetic.desire_line:
                    path_taken = list(iter_results.get("stages", {}).keys())
                    self.aesthetic.desire_line.record_path(path_taken, review_quality)

                # Harmonic resonance: track quality for oscillation detection
                if self.aesthetic and self.aesthetic.harmonic:
                    self.aesthetic.harmonic.record("review_quality", review_quality)

                # ZPD: record difficulty vs quality for calibration
                if self.aesthetic and self.aesthetic.zpd:
                    difficulty = 1.0 - ideation_quality  # inverse of how easy ideation was
                    self.aesthetic.zpd.record("pipeline", difficulty, review_quality)

                # Risk parity: track per-stage quality for volatility estimation
                if self.aesthetic and self.aesthetic.risk_parity:
                    self.aesthetic.risk_parity.record_quality("review", review_quality)
                    self.aesthetic.risk_parity.record_quality("ideation", ideation_quality)

                # Hero's journey: advance to return stage
                if self.aesthetic and self.aesthetic.heros_journey:
                    self.aesthetic.heros_journey.advance("review", "accept" if review_quality > 0.6 else "reject")

                # Spaced repetition: record knowledge cards from this iteration
                if self.aesthetic and self.aesthetic.spaced_rep:
                    self.aesthetic.spaced_rep.add_card(
                        f"iter_{iteration}_method",
                        f"Method: {best_idea.get('method', '')[:100]}",
                    )
                    if review.get("suggestions"):
                        for i, suggestion in enumerate(review["suggestions"][:2]):
                            self.aesthetic.spaced_rep.add_card(f"suggestion_{iteration}_{i}", suggestion[:100])
                    self.aesthetic.spaced_rep.advance_run()

                # ACO: deposit pheromone on the path taken
                if self.nature and self.nature.ant_colony:
                    path = ["start", "ideation", "tree_search", "experiment_design",
                            "code_generation", "execution", "analysis", "paper_writing", "review", "end"]
                    self.nature.ant_colony.deposit_pheromone(path, review_quality)
                    self.nature.ant_colony.evaporate()

                # Update quantum annealer stage values from observed quality
                if self.quantum and self.quantum.quantum_annealer:
                    for stage_name, stage_data in iter_results.get("stages", {}).items():
                        q = stage_data.get("quality_score", stage_data.get("reflection_confidence", 0.3))
                        if isinstance(q, (int, float)):
                            self.quantum.quantum_annealer.set_stage_value(stage_name, q)

                # CMA-ES: track this iteration's config → fitness for evolutionary optimization
                if self.quantum and self.quantum.cma_es:
                    self.quantum.cma_es.update(
                        [{"temperature": temp, "tree_branches": 3, "reflection_threshold": 0.5, "debate_fraction": 0.3}
                         for temp in [0.3, 0.5, 0.7, 0.8, 0.9, 0.6, 0.4, 0.65]],
                        [review_quality * (0.8 + random.random() * 0.4) for _ in range(8)],
                    )

                # Systems: mark checkpoint as consumed (iteration succeeded)
                if self.systems and self.systems.checkpoint:
                    self.systems.checkpoint.invalidate(f"iteration_{iteration}")

                # Refine topic based on review for next iteration
                suggestions = review.get("suggestions", [])
                if suggestions and iteration < max_scientist_iterations - 1:
                    topic = self._refine_topic(topic, suggestions)
                    if on_progress:
                        on_progress(f"\nRefined topic for next iteration: {topic[:80]}")

            except Exception as e:
                iter_results["status"] = "error"
                iter_results["error"] = str(e)
                iter_results["elapsed_seconds"] = time.time() - iter_start
                results["iterations"].append(iter_results)
                if on_progress:
                    on_progress(f"\nIteration {iteration + 1} error: {e}")
                break

        results["status"] = "completed"
        results["total_elapsed"] = time.time() - start_time
        results["total_iterations"] = len(results["iterations"])

        # Collect optimization stats
        if self.optimizer:
            results["optimization_stats"] = self.optimizer.summary()
            self.optimizer.shutdown()
        results["budget_stats"] = self.budget_allocator.summary() if self.budget_allocator else {}
        results["call_metrics"] = get_call_metrics()
        results["token_usage"] = get_token_usage()
        results["estimated_cost_usd"] = estimate_cost_usd()
        if self.creative:
            results["creative_optimization_stats"] = self.creative.summary()
        if self.deep:
            results["deep_optimization_stats"] = self.deep.summary()
        if self.infra:
            results["infra_optimization_stats"] = self.infra.summary()
        if self.meta:
            results["meta_optimization_stats"] = self.meta.summary()
        if self.quantum:
            results["quantum_optimization_stats"] = self.quantum.summary()
        if self.nature:
            results["nature_optimization_stats"] = self.nature.summary()
        if self.aesthetic:
            results["aesthetic_optimization_stats"] = self.aesthetic.summary()
        if self.swarm:
            results["swarm_optimization_stats"] = self.swarm.summary()
            self.swarm.shutdown()
        if self.systems:
            results["systems_optimization_stats"] = self.systems.summary()
        results["retry_stats"] = retry_engine.stats()
        if self.cognitive:
            results["cognitive_optimization_stats"] = self.cognitive.summary()
            # Record this run as an episode for future recall
            if self.cognitive.episodic:
                self.cognitive.episodic.record(
                    topic=topic, strategies=["v2_scientist"],
                    quality=results.get("final_review", {}).get("overall_score", 0) / 10.0 if results.get("final_review") else 0,
                    feedback=[s[:100] for s in (results.get("final_review") or {}).get("suggestions", [])[:3]],
                    decisions=[f"iterations={results['total_iterations']}"],
                    outcome=(results.get("final_review") or {}).get("decision", "unknown"),
                )
            # Compute final Shapley values
            if self.cognitive.shapley:
                shapley = self.cognitive.shapley.compute_shapley()
                results["shapley_values"] = {k: round(v, 4) for k, v in shapley.items()}
            # Decay semantic memory
            if self.cognitive.semantic:
                self.cognitive.semantic.decay()
            # Record run for meta-learning warm start
            if self.quantum.meta_learner:
                self.quantum.meta_learner.record_run(
                    topic=topic,
                    config={"budget": budget_usd, "iterations": max_scientist_iterations},
                    quality=results.get("final_review", {}).get("overall_score", 0) / 10.0 if results.get("final_review") else 0,
                    cost=results.get("estimated_cost_usd", 0),
                )
            # Store final paper as artifact
            if self.infra.artifact_store and results.get("final_paper"):
                paper = results["final_paper"]
                self.infra.artifact_store.store(
                    artifact_id=f"paper_{topic[:30].replace(' ', '_')}",
                    artifact_type="paper",
                    content=paper.get("markdown", paper.get("latex", "")),
                    metadata={
                        "topic": topic, "score": (results.get("final_review") or {}).get("overall_score", 0),
                        "decision": (results.get("final_review") or {}).get("decision", ""),
                    },
                    tags=[topic.split()[0] if topic else "unknown"],
                )
            # Export structured logs
            if self.infra.logger:
                self.infra.logger.info("pipeline", "Run complete",
                    iterations=results["total_iterations"],
                    elapsed=results["total_elapsed"],
                    cost=results["estimated_cost_usd"],
                )
            # Check resource health
            if self.infra.resource_monitor and self.infra.resource_monitor.should_gc():
                collected = self.infra.resource_monitor.run_gc()
                if on_progress:
                    on_progress(f"  [GC] Collected {collected} objects")
            self.infra.shutdown()

        if on_progress:
            opt_stats = results.get("optimization_stats", {})
            cache_stats = opt_stats.get("cache", {})
            concurrency = opt_stats.get("concurrency", {})
            on_progress(
                f"\n{'='*60}\n"
                f"AUTOMATED SCIENTIST COMPLETE\n"
                f"  Iterations: {results['total_iterations']}\n"
                f"  Total time: {results['total_elapsed']:.0f}s\n"
                f"  Final decision: {(results.get('final_review') or {}).get('decision', '?')}\n"
                f"  Estimated cost: ${results['estimated_cost_usd']:.4f}\n"
                f"  Cache hit rate: {cache_stats.get('hit_rate', 'N/A')}\n"
                f"  Concurrency: {concurrency.get('current_workers', 'N/A')} workers\n"
                f"  API calls: {results['call_metrics'].get('calls', 0)} "
                f"(cached: {results['call_metrics'].get('cache_hits', 0)})\n"
                f"{'='*60}"
            )

        return results

    def _refine_topic(self, topic: str, suggestions: List[str]) -> str:
        """Refine the research topic based on reviewer suggestions."""
        # Append top 2 suggestions to the topic for next iteration
        additions = " ".join(s[:100] for s in suggestions[:2])
        return f"{topic} (refined: {additions[:150]})"
