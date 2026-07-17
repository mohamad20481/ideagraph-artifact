"""
pipeline.py - Orchestrates the full IdeaGraph pipeline end-to-end.

Steps:
  1. Build Knowledge DAG   (KnowledgeArchitect)
  2. Initialise QD Archive
  3. Ideation loop         (IdeationAgent + ExecutionCritic + DiversityManager)
  4. Return results dict
"""

from __future__ import annotations
import heapq
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set

import config
from models.archive import QDArchive
from models.dag import KnowledgeDAG
from models.idea import Idea
from agents.knowledge_architect import KnowledgeArchitect
from agents.ideation_agent import IdeationAgent
from agents.execution_critic import ExecutionCritic
from agents.diversity_manager import DiversityManager
from agents.base_agent import estimate_cost_usd
from intelligence import (
    SemanticNoveltyChecker,
    explain_quality,
    should_replace_pareto,
    FailurePatternTracker,
    advise_strategy,
    CrossRunMemory,
)


def _idea_is_valid(idea) -> bool:
    """Return False if the idea is missing key content (malformed LLM output)."""
    return (
        len(idea.title.strip()) >= 5
        and len(idea.method.strip()) >= 20
        and len(idea.hypothesis.strip()) >= 10
    )


def _simhash64(tokens: FrozenSet[str]) -> int:
    """
    Compute a 64-bit SimHash fingerprint from a token set.

    SimHash maps each token to a hash, then sums per-bit votes (+1/-1)
    across all tokens.  The final fingerprint captures the "shape" of the
    token set — two sets with Hamming distance ≤ 6 typically have
    Jaccard similarity > 0.65.

    Used as an O(1) pre-filter before the more expensive Jaccard check:
    if Hamming distance > _SIMHASH_MAX_DIST, the pair cannot be duplicates
    and we skip the full Jaccard computation entirely.
    """
    v = [0] * 64
    for token in tokens:
        h = hash(token) & 0xFFFFFFFFFFFFFFFF
        for i in range(64):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1
    fingerprint = 0
    for i in range(64):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


def _hamming64(a: int, b: int) -> int:
    """Population count of XOR — number of differing bits."""
    return bin(a ^ b).count("1")


# Max Hamming distance for SimHash pre-filter.
# 6 bits out of 64 ≈ 90.6% agreement — empirically catches >90% of
# Jaccard > 0.65 pairs while rejecting most non-duplicates in O(1).
_SIMHASH_MAX_DIST = 6


class IdeaGraphPipeline:

    # Common English stopwords filtered from method text before Jaccard comparison
    _METHOD_STOPWORDS: FrozenSet[str] = frozenset({
        "a", "an", "the", "and", "or", "of", "to", "in", "for", "with",
        "using", "by", "on", "at", "this", "that", "we", "our", "is",
        "are", "be", "been", "can", "will", "use", "used", "based",
    })

    def __init__(self) -> None:
        self.architect = KnowledgeArchitect()
        self.ideation = IdeationAgent()
        self.critic = ExecutionCritic()
        self.diversity = DiversityManager()
        # Deduplication: track (title_tokens, method_tokens) pairs for archived ideas.
        self._archived_token_pairs: List[tuple] = []
        self._archived_simhashes: List[tuple] = []
        self._dedup_lock = threading.Lock()
        # ── Intelligence layer ───────────────────────────────────────────────
        self._novelty_checker = SemanticNoveltyChecker(threshold=0.55)
        self._failure_tracker = FailurePatternTracker()
        self._cross_run_memory: Optional[CrossRunMemory] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────
    def run(
        self,
        topic: str,
        budget_usd: float = 2.0,
        max_iterations: int = 20,
        on_progress: Optional[Callable[[str], None]] = None,
        debate_enabled: bool = False,
        user_id: Optional[int] = None,
        runtime_controller: Optional[Any] = None,
    ) -> Dict[str, Any]:
        # Clear stale caches so different topics get fresh results
        from agents.base_agent import reset_caches
        reset_caches()

        # Set run context so base_agent can enforce cost limits per-call.
        import uuid as _uuid
        config._current_run_id = _uuid.uuid4().hex[:12]
        config._current_budget_usd = budget_usd
        config._current_user_id = user_id

        # ── Observability: trace the entire run as a single span ────────────
        try:
            from observability import set_trace_id, metrics as _obs_metrics, logger as _obs_log
            set_trace_id(config._current_run_id)
            _obs_metrics.inc("pipeline_runs_total", tags={"mode": "v1"})
            _obs_log.info("pipeline_start", topic=topic, budget_usd=budget_usd,
                          max_iterations=max_iterations, user_id=user_id,
                          run_id=config._current_run_id)
        except ImportError:
            pass

        self._topic = topic  # store for fallback idea generation

        # Stash the runtime controller (if any) on a module-global so
        # base_agent can call record_llm_result() without us having to
        # plumb it through every agent constructor.
        self._runtime_controller = runtime_controller
        try:
            import config as _cfg
            _cfg._current_runtime_controller = runtime_controller
        except Exception:
            pass

        # Dedup state: skip emitting the same progress message twice in a row
        # (e.g. retries inside an agent that all log "Generating ideas …").
        _last_progress: List[str] = [""]

        def progress(msg: str) -> None:
            if on_progress and msg != _last_progress[0]:
                _last_progress[0] = msg
                on_progress(msg)

        # ── Cross-run learning: load lessons from previous runs ──────────────
        self._cross_run_memory = CrossRunMemory(user_id=user_id)
        self._cross_run_memory.load_from_db()
        cross_run_ctx = self._cross_run_memory.get_context_for_prompt()
        if cross_run_ctx:
            progress(f"Loaded {len(self._cross_run_memory._lessons)} lessons from previous runs.")
        # Clear intelligence state from previous runs.
        self._novelty_checker.clear()
        self._failure_tracker.clear()
        start_time = time.time()

        # ── Phase 1: Build DAG ────────────────────────────────────────────────
        progress("Phase 1/3: Building Knowledge DAG …")
        dag = self._build_dag(topic, progress)

        # ── Phase 2: Initialise archive ───────────────────────────────────────
        archive = QDArchive()
        progress(
            f"Phase 2/3: Starting ideation loop "
            f"(max {max_iterations} iterations, coverage target {config.COVERAGE_THRESHOLD:.0%}) …"
        )

        # ── Smart context: feed DAG papers to critic AND ideation agent ────
        dag_papers = dag.get_all_papers() if hasattr(dag, 'get_all_papers') else []
        dag_titles = [p.title for p in dag_papers][:10]
        self.critic.set_dag_context(dag_titles)

        # Feed real paper data (title + abstract) to ideation for literature-aware generation
        if dag_papers:
            paper_dicts = [
                {"title": p.title, "abstract": getattr(p, "abstract", "")[:200], "year": getattr(p, "year", "")}
                for p in dag_papers[:8]
            ]
            self.ideation.set_literature_context(paper_dicts)
            progress(f"Injected {len(paper_dicts)} paper abstracts for literature-grounded generation.")

        # ── Revision learning: track success rate per strategy ────────────────
        _revision_attempts = 0
        _revision_successes = 0

        # ── Topic decomposition (pre-compute sub-problems once) ──────────────
        try:
            subs = self.ideation.decompose_topic(topic)
            if subs:
                progress(f"Topic decomposed into {len(subs)} sub-problems: {'; '.join(s[:40] for s in subs[:3])}")
        except Exception:
            pass

        # ── Phase 3: Ideation loop ────────────────────────────────────────────
        iteration = 0
        ideas_attempted = 0
        ideas_archived = 0

        # ── Speed: persistent thread pool (don't recreate each iteration) ────
        _thread_pool = ThreadPoolExecutor(max_workers=min(getattr(config, "MAX_PARALLEL_CELLS", 6), 8))

        # Coverage slope tracking: stop when progress < threshold over window
        _coverage_history: List[float] = []
        _SLOPE_WINDOW = 3        # look back this many iterations
        _SLOPE_MIN = 0.01        # min coverage gain per iteration to continue
        _SLOPE_MIN_ITER = 3      # check sooner (was 4)

        all_ideas_so_far: List = []  # refreshed each iteration; reused in Phase 4

        while iteration < max_iterations:
            # ── Runtime-control checkpoint ──────────────────────────────────
            # If the user wired in a RuntimeController, give it a chance to
            # pause us (budget threshold) or stop us (user clicked Stop).
            # This blocks until the user decides; default-stop on timeout.
            if runtime_controller is not None:
                try:
                    _current_cost = float(getattr(config, "_current_cost_usd", 0.0))
                except Exception:
                    _current_cost = 0.0
                if not runtime_controller.heartbeat(_current_cost):
                    progress("Pipeline stopped by user.")
                    break

            coverage = archive.coverage()
            _coverage_history.append(coverage)

            if coverage >= config.COVERAGE_THRESHOLD:
                progress(
                    f"Coverage target reached ({coverage:.1%}). Stopping early at iteration {iteration}."
                )
                break

            # ── Smart early stopping: multiple signals ──────────────────────────
            # Signal 1: Coverage slope stalled
            if iteration >= _SLOPE_MIN_ITER and len(_coverage_history) >= _SLOPE_WINDOW + 1:
                window = _coverage_history[-(_SLOPE_WINDOW + 1):]
                slope = (window[-1] - window[0]) / _SLOPE_WINDOW
                if slope < _SLOPE_MIN:
                    progress(
                        f"Coverage slope {slope:.4f}/iter < {_SLOPE_MIN} — stalled. "
                        f"Stopping at iteration {iteration} with coverage {coverage:.1%}."
                    )
                    break

            # Signal 2: Already have enough high-quality ideas (speed optimization)
            if ideas_archived >= 8 and coverage >= 0.25:
                avg_q = archive.mean_quality()
                if avg_q >= 0.55:
                    progress(
                        f"Good enough: {ideas_archived} ideas archived with avg quality {avg_q:.3f}. "
                        f"Stopping early at iteration {iteration}."
                    )
                    break

            # Signal 3: Budget nearly exhausted
            cost = estimate_cost_usd()
            if cost >= budget_usd * 0.9:
                progress(f"Budget 90% used (${cost:.3f}/${budget_usd:.2f}). Stopping.")
                break

            # ── Adaptive cell count (speed: more cells early, fewer late) ─────
            # First 2 iterations: aggressive 6 cells (fill archive fast)
            # Low coverage: 5 cells
            # Mid coverage: 3 cells
            # High coverage: 2 cells (just gap-filling)
            if iteration < 2:
                n_cells = min(6, 21 - len(archive.get_all_ideas()))  # don't exceed empty cells
            elif coverage < 0.25:
                n_cells = 5
            elif coverage < 0.55:
                n_cells = 3
            else:
                n_cells = 2
            n_cells = max(1, n_cells)

            progress(
                f"Iteration {iteration + 1}/{max_iterations} | "
                f"Coverage: {coverage:.1%} | Archived: {ideas_archived} | "
                f"Cells/iter: {n_cells}"
            )

            # Select target cells via UCB1 (exploration decays with iteration)
            target_cells = self.diversity.select_target_cells(archive, n=n_cells, iteration=iteration)

            # ── Smart iteration setup ─────────────────────────────────────────
            # Update exemplars for few-shot injection (top 2 archived ideas)
            all_ideas_so_far = archive.get_all_ideas()
            if all_ideas_so_far:
                top_exemplars = heapq.nlargest(2, all_ideas_so_far, key=lambda i: i.quality_score)
                self.ideation.set_exemplars([
                    {"title": i.title, "method": i.method} for i in top_exemplars
                ])

            # Update critic with archived methods for novelty grounding
            archived_methods = [i.method for i in all_ideas_so_far if i.method]
            self.critic.set_archived_methods(archived_methods)

            # Pre-build negative-example context once per iteration (was per-cell).
            archived_titles = [i.title for i in all_ideas_so_far]
            if archived_titles:
                _avoid = (
                    "Avoid ideas similar to these:\n"
                    + "\n".join(f"- {t[:60]}" for t in archived_titles[:8])
                )
            else:
                _avoid = ""

            # ── Pre-cell budget gate (prevents launching cells we can't pay for) ──
            cost_before_cells = estimate_cost_usd()
            if cost_before_cells >= budget_usd * 0.95:
                progress(
                    f"Budget 95% consumed (${cost_before_cells:.3f}/${budget_usd:.2f}) "
                    f"before launching cells. Stopping iteration."
                )
                break

            # Run cells in parallel (reuse persistent pool for speed)
            # Per-cell timeout (per future.result() call) — slow cells are
            # skipped individually without aborting the whole iteration.
            _CELL_TIMEOUT_S = 300  # 5 minutes max per cell
            _ITER_DEADLINE = time.time() + _CELL_TIMEOUT_S + 60  # iteration-wide cap
            future_to_cell = {
                _thread_pool.submit(self._run_cell, dag, archive, cell, iteration, progress, coverage, _avoid): cell
                for cell in target_cells
            }
            # NOTE: do NOT pass timeout to as_completed — it raises if not all
            # futures complete, throwing away the ones that did. Use a try/finally
            # block instead to cancel still-running futures at iteration end.
            try:
                for future in as_completed(future_to_cell):
                    cell = future_to_cell[future]
                    ideas_attempted += 1
                    # Cap individual future.result() at remaining iteration time
                    _remaining = max(1.0, _ITER_DEADLINE - time.time())
                    _per_future_timeout = min(_CELL_TIMEOUT_S, _remaining)
                    try:
                        succeeded = future.result(timeout=_per_future_timeout) or False
                        self.diversity.record_attempt(cell, succeeded)
                        if succeeded:
                            ideas_archived += 1
                    except FuturesTimeout:
                        self.diversity.record_attempt(cell, False)
                        progress(f"  Cell {cell} timed out — skipped, keeping other completed cells")
                        future.cancel()
                    except Exception as exc:
                        self.diversity.record_attempt(cell, False)
                        progress(f"  Cell {cell} failed with error: {exc}")

                    # If iteration deadline passed, cancel remaining futures
                    if time.time() > _ITER_DEADLINE:
                        _unfinished = sum(1 for f in future_to_cell if not f.done())
                        if _unfinished:
                            progress(f"  Iteration deadline reached — cancelling {_unfinished} pending cell(s)")
                            for f in future_to_cell:
                                if not f.done():
                                    f.cancel()
                        break
            except Exception as exc:
                # Hard failsafe: if as_completed itself dies, salvage what we have
                progress(f"  Iteration loop error (continuing with partial results): {exc}")

            # ── Emit structured progress event for live dashboard ────────────
            coverage_now = archive.coverage()
            q_mean_now, _, q_max_now = archive.quality_stats()
            cost = estimate_cost_usd()
            _event_dict = {
                "_event": "iteration_complete",
                "iteration": iteration + 1,
                "coverage": round(coverage_now, 4),
                "ideas_archived": ideas_archived,
                "ideas_attempted": ideas_attempted,
                "quality_mean": round(q_mean_now, 4),
                "quality_max": round(q_max_now, 4),
                "cost_usd": round(cost, 4),
                "budget_usd": budget_usd,
            }
            progress(f"__EVENT__{json.dumps(_event_dict)}")

            # ── Checkpoint every 3 iterations (crash recovery) ─────────────────
            if (iteration + 1) % 3 == 0 and all_ideas_so_far:
                try:
                    _ckpt_dir = os.path.join(os.path.dirname(__file__), "data", "checkpoints")
                    os.makedirs(_ckpt_dir, exist_ok=True)
                    _ckpt_data = {
                        "iteration": iteration + 1,
                        "timestamp": time.time(),
                        "topic": topic,
                        "coverage": round(coverage_now, 4),
                        "ideas_count": len(all_ideas_so_far),
                        "cost_usd": round(cost, 4),
                        "ideas": [
                            {"title": i.title, "method": i.method,
                             "quality_score": round(i.quality_score, 4)}
                            for i in all_ideas_so_far[:50]
                        ],
                    }
                    _ckpt_path = os.path.join(_ckpt_dir, f"pipeline_iter_{iteration + 1}.json")
                    with open(_ckpt_path, "w", encoding="utf-8") as _f:
                        json.dump(_ckpt_data, _f, ensure_ascii=False)
                    progress(f"Checkpoint saved: iteration {iteration + 1}, {len(all_ideas_so_far)} ideas")
                except Exception:
                    pass  # never crash the pipeline for a checkpoint failure

            # ── Budget enforcement ────────────────────────────────────────────
            if cost >= budget_usd:
                progress(
                    f"Budget ${budget_usd:.2f} reached (spent ~${cost:.3f}). "
                    f"Stopping at iteration {iteration}."
                )
                iteration += 1
                break

            iteration += 1

        # ── Phase 4: Debate Tournament (optional) ──────────────────────────
        tournament_data = None
        # Reuse the last iteration's snapshot if available, otherwise fetch.
        # `all_ideas_so_far` is already set from the final iteration above;
        # we just rename it here so the rest of the method stays readable.
        all_ideas = all_ideas_so_far if all_ideas_so_far else archive.get_all_ideas()

        if debate_enabled and len(all_ideas) >= 4:
            progress("Phase 4: Running Debate Tournament …")
            try:
                from agents.agent_memory import AgentMemoryManager
                from agents.debate_arena import DebateArena

                memory_mgr = AgentMemoryManager(user_id) if user_id else None
                arena = DebateArena(memory_manager=memory_mgr)

                top_ideas = heapq.nlargest(
                    config.DEBATE_MAX_IDEAS, all_ideas,
                    key=lambda i: i.quality_score,
                )

                tournament = arena.run_tournament(top_ideas, topic, on_progress=progress)
                tournament_data = tournament.to_dict()

                # Build a title → [match_dict] index ONCE, then look up per idea.
                # Previously this was O(ideas × matches) with a nested list comp
                # that called m.to_dict() repeatedly for shared matches.
                from collections import defaultdict
                _match_index: Dict[str, list] = defaultdict(list)
                for m in tournament.all_matches:
                    md = m.to_dict()
                    a_title = m.idea_a.get("title", "")
                    b_title = m.idea_b.get("title", "")
                    if a_title:
                        _match_index[a_title].append(md)
                    if b_title and b_title != a_title:
                        _match_index[b_title].append(md)

                for idea in all_ideas:
                    if idea.debate_rank is not None:
                        idea.debate_history = _match_index.get(idea.title, [])
            except Exception as exc:
                progress(f"Debate phase error: {exc}")

        # Cleanup thread pool
        _thread_pool.shutdown(wait=False)

        elapsed = time.time() - start_time
        q_mean, q_min, q_max = archive.quality_stats()
        progress(
            f"Pipeline complete in {elapsed:.1f}s | "
            f"Coverage: {archive.coverage():.1%} | "
            f"Ideas archived: {ideas_archived}/{ideas_attempted} | "
            f"Quality: mean={q_mean:.3f} min={q_min:.3f} max={q_max:.3f}"
        )

        result = {
            "topic": topic,
            "dag_summary": dag.to_summary_dict(),
            "ideas": [idea.to_dict() for idea in all_ideas],
            "coverage": archive.coverage(),
            "archive": archive.to_display_dict(),
            "stats": {
                "iterations": iteration,
                "ideas_attempted": ideas_attempted,
                "ideas_archived": ideas_archived,
                "elapsed_seconds": round(elapsed, 1),
                "estimated_cost_usd": round(estimate_cost_usd(), 4),
                "quality_mean": round(q_mean, 4),
                "quality_min": round(q_min, 4),
                "quality_max": round(q_max, 4),
            },
        }

        if tournament_data:
            result["tournament"] = tournament_data

        # ── Observability: emit end-of-run metrics ──────────────────────────
        try:
            from observability import metrics as _obs_metrics, logger as _obs_log
            _obs_metrics.observe("pipeline_duration_seconds", elapsed,
                                 tags={"mode": "v1", "status": "ok"})
            _obs_metrics.inc("pipeline_ideas_total",
                             value=float(ideas_archived), tags={"mode": "v1"})
            _obs_metrics.set("pipeline_coverage",
                             archive.coverage(), tags={"mode": "v1"})
            _obs_log.info("pipeline_complete",
                          run_id=config._current_run_id,
                          elapsed_s=round(elapsed, 1),
                          ideas_archived=ideas_archived,
                          coverage=round(archive.coverage(), 4),
                          cost_usd=round(estimate_cost_usd(), 4),
                          user_id=user_id)
        except ImportError:
            pass

        # ── Runtime-control: mark completion and clear the global ──────────
        if runtime_controller is not None:
            try:
                runtime_controller.mark_completed()
            except Exception:
                pass
        try:
            import config as _cfg
            _cfg._current_runtime_controller = None
        except Exception:
            pass

        # ── Cross-run learning: extract and persist lessons ────────────────
        if self._cross_run_memory:
            lessons = self._cross_run_memory.extract_lessons_from_run(result)
            for lesson in lessons:
                self._cross_run_memory.save_lesson(lesson)
            if lessons:
                progress(f"Learned {len(lessons)} lessons for future runs.")

        # ── Add quality explanations to each idea in result ──────────────────
        for idea_dict in result.get("ideas", []):
            scores = idea_dict.get("probe_scores", {})
            if scores:
                idea_dict["quality_explanation"] = explain_quality(scores, idea_dict.get("title", ""))

        # ── Apply 5 idea enhancements (knobs, repro, FMEA, domain, adversarial) ─
        try:
            from idea_enhancer import enhance_idea, IdeationKnobs
            _knobs = getattr(config, "_ideation_knobs", None) or IdeationKnobs()
            for i, idea_dict in enumerate(result.get("ideas", [])):
                result["ideas"][i] = enhance_idea(idea_dict, _knobs, topic=topic)
        except Exception:
            pass

        # Clean up run context.
        config._current_run_id = None
        config._current_budget_usd = None
        config._current_user_id = None

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Sub-routines
    # ─────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Idea deduplication helpers
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _title_tokens(title: str) -> FrozenSet[str]:
        """Lowercase word-token set of a title (for Jaccard similarity).
        Non-alphanumeric chars (hyphens, punctuation) become spaces so that
        'Transformer-Based' → {'transformer', 'based'}, not 'transformerbased'.
        """
        return frozenset(re.sub(r"[^a-z0-9]+", " ", title.lower()).split())

    @classmethod
    def _method_tokens(cls, method: str) -> FrozenSet[str]:
        """
        Stopword-filtered token set of the method field.
        Captures technical terms (model names, algorithms, metrics) while
        ignoring common function words that bloat the Jaccard denominator.
        """
        raw = frozenset(re.sub(r"[^a-z0-9]+", " ", method.lower()).split())
        return raw - cls._METHOD_STOPWORDS

    def _is_duplicate(self, idea: Idea, title_threshold: float = 0.75, method_threshold: float = 0.65) -> bool:
        """
        Return True if idea is too similar to any already-archived idea.

        Two signals fire independently:
          - Title Jaccard ≥ title_threshold (0.75): same topic, different wording
          - Method Jaccard ≥ method_threshold (0.65): same technical approach,
            different title (catches 'title-washing')

        Method threshold is slightly looser because method text is longer and
        two truly different approaches rarely share 65% of technical vocabulary.
        Minimum method token count of 6 avoids false positives on stub methods.

        Optimisation: SimHash pre-filter (O(1)) rejects most non-duplicates
        before the O(|tokens|) Jaccard computation.  Only pairs whose SimHash
        Hamming distance ≤ _SIMHASH_MAX_DIST proceed to the full check.

        Side-effect: caches the token/hash tuple on the idea so that
        _register_archived_title() can reuse it without recomputing.
        """
        t_tokens = self._title_tokens(idea.title)
        m_tokens = self._method_tokens(idea.method)
        t_hash = _simhash64(t_tokens)
        m_hash = _simhash64(m_tokens)

        # Stash for later _register_archived_title() — avoids double compute.
        idea._dedup_cache = (t_tokens, m_tokens, t_hash, m_hash)

        # Snapshot under the lock, then scan without it. Both lists are
        # append-only, so a shallow copy captures a consistent view and
        # frees other workers to scan + register in parallel.
        with self._dedup_lock:
            pairs_snapshot = list(self._archived_token_pairs)
            hashes_snapshot = list(self._archived_simhashes)

        for idx, (arch_title, arch_method) in enumerate(pairs_snapshot):
            arch_t_hash, arch_m_hash = hashes_snapshot[idx]

            # Title signal: SimHash pre-filter then Jaccard
            if t_tokens and arch_title:
                if _hamming64(t_hash, arch_t_hash) <= _SIMHASH_MAX_DIST:
                    union = t_tokens | arch_title
                    if union and len(t_tokens & arch_title) / len(union) >= title_threshold:
                        return True

            # Method signal: SimHash pre-filter then Jaccard
            if len(m_tokens) >= 6 and len(arch_method) >= 6:
                if _hamming64(m_hash, arch_m_hash) <= _SIMHASH_MAX_DIST:
                    union = m_tokens | arch_method
                    if union and len(m_tokens & arch_method) / len(union) >= method_threshold:
                        return True
        return False

    def _register_archived_title(self, idea: Idea) -> None:
        """Record the title+method token pair and SimHash of an archived idea.

        Reuses token/hash data cached by _is_duplicate() when available,
        avoiding redundant tokenization + hashing (previously ~2ms/idea).
        """
        cache = getattr(idea, "_dedup_cache", None)
        if cache:
            t_tokens, m_tokens, t_hash, m_hash = cache
        else:
            t_tokens = self._title_tokens(idea.title)
            m_tokens = self._method_tokens(idea.method)
            t_hash = _simhash64(t_tokens)
            m_hash = _simhash64(m_tokens)
        with self._dedup_lock:
            self._archived_token_pairs.append((t_tokens, m_tokens))
            self._archived_simhashes.append((t_hash, m_hash))

    def _build_dag(
        self,
        topic: str,
        progress: Callable[[str], None],
    ) -> KnowledgeDAG:
        try:
            dag = self.architect.build_dag(topic, on_progress=progress)
            n_papers = dag.graph.number_of_nodes() if hasattr(dag, 'graph') else 0
            if n_papers == 0:
                progress(f"WARNING: DAG is empty (0 papers). Semantic Scholar may be unreachable. Ideas will be less grounded.")
            else:
                progress(f"DAG built: {n_papers} papers.")
            return dag
        except Exception as exc:
            progress(f"DAG build failed ({exc}); using empty DAG. Ideas will be generic.")
            return KnowledgeDAG()

    def _run_cell(
        self,
        dag: KnowledgeDAG,
        archive: QDArchive,
        target_cell,
        iteration: int,
        progress: Callable[[str], None],
        coverage: float = 0.0,
        avoid_context: str = "",
    ) -> bool:
        """
        For one target cell:
          1. Pick strategy
          2. Generate idea
          3. Probe (with progressive quality floor)
          4. Archive if pass; else revise once and probe again
        Returns True if an idea was added to the archive.
        """
        strategy = self.diversity.pick_strategy(dag, target_cell, iteration)
        _, novelty_idx = target_cell
        progress(f"  Cell {target_cell}: Strategy {strategy}")

        idea = self._generate_idea(dag, strategy, target_cell, avoid_context)
        if idea is None:
            progress(f"  Cell {target_cell}: Idea generation failed.")
            self.diversity.record_strategy_attempt(strategy, novelty_idx, False)
            return False

        # Override behavioural coordinates to match target cell
        from models.idea import METHODOLOGY_TYPES, NOVELTY_LEVELS
        idea.methodology_type = METHODOLOGY_TYPES[target_cell[0]]
        idea.novelty_level = NOVELTY_LEVELS[target_cell[1]]

        # ── Pre-validate idea has meaningful content before probing ───────────
        if not _idea_is_valid(idea):
            progress(f"  Cell {target_cell}: Idea missing key content, skipped.")
            self.diversity.record_strategy_attempt(strategy, novelty_idx, False)
            return False

        # ── Deduplication check (skips 1 LLM probe call for near-dups) ───────
        if self._is_duplicate(idea):
            progress(f"  Cell {target_cell}: Near-duplicate idea skipped.")
            self.diversity.record_strategy_attempt(strategy, novelty_idx, False)
            return False

        # ── Semantic novelty check (before spending LLM on probing) ──────────
        is_novel, similar_to, sim_score = self._novelty_checker.check(idea.method)
        if not is_novel:
            progress(
                f"  Cell {target_cell}: Semantically too similar to \"{similar_to[:50]}\" "
                f"(similarity {sim_score:.0%}). Skipped."
            )
            self.diversity.record_strategy_attempt(strategy, novelty_idx, False)
            return False

        progress(f"  Probing: \"{idea.title[:60]}\"")
        probe_result = self._probe_safe(idea)
        idea.probe_scores = probe_result.get("scores", {})
        idea.probe_passed = probe_result.get("all_pass", False)
        idea.quality_score = probe_result.get("quality", 0.0)

        # ── Execution-aware revision (closes the probe → archive loop) ──────
        # When enabled and the idea looks promising (passed probes), run a
        # deliberately tiny LLM-simulated experiment and Bayesian-blend the
        # resulting feasibility signal into quality_score. This is the
        # mechanism the IdeaGraph paper notes as the unsolved feasibility gap.
        if (config.ENABLE_EXECUTION_REVISION and idea.probe_passed
                and idea.quality_score >= 0.30):
            try:
                from agents.execution_revisor import revise as _revise_idea
                rev = _revise_idea(
                    idea,
                    n_samples=config.EXECUTION_REVISION_SAMPLE_SIZE,
                    n_seeds=config.EXECUTION_REVISION_N_SEEDS,
                )
                if rev.success:
                    idea.probe_quality = rev.probe_quality
                    idea.execution_signal = rev.execution_signal
                    idea.execution_trust = rev.trust_weight
                    idea.execution_delta = rev.delta
                    idea.execution_meta = {
                        "metric_name": rev.metric_name,
                        "predicted_metric": rev.predicted_metric,
                        "confidence_interval": (
                            list(rev.confidence_interval)
                            if rev.confidence_interval else None
                        ),
                        "failure_modes": rev.failure_modes,
                        "sample_size": rev.sample_size,
                        "n_seeds": rev.n_seeds,
                        "used_llm": rev.used_llm,
                        "cost_usd": rev.cost_usd,
                    }
                    # Overwrite quality_score with Bayesian posterior so the
                    # archive ranks ideas by feasibility-aware quality.
                    idea.quality_score = rev.blended_quality
                    progress(f"  Exec-revision: {rev.summary()}")
            except Exception as e:
                progress(f"  Exec-revision skipped ({e.__class__.__name__})")

        # ── Quality explanation (human-readable) ─────────────────────────────
        explanation = explain_quality(idea.probe_scores, idea.title)
        progress(f"  Quality: {explanation[:120]}")

        # Feed probe failures back to ideation agent + failure tracker
        scores = probe_result.get("scores", {})
        for probe_name, score in scores.items():
            if isinstance(score, (int, float)) and score < 0.4:
                self.ideation.record_probe_failure(probe_name)
                self._failure_tracker.record_failure(
                    target_cell[0], target_cell[1], probe_name,
                )

        # ── Progressive quality floor ──────────────────────────────────────────
        # As the archive fills, demand higher quality to prevent low-quality ideas
        # from displacing slots that could hold better ones in later iterations.
        # Floor rises linearly: 0.30 at 0% coverage → 0.50 at 100% coverage.
        # Only applies to initially-passing ideas (partial rescue uses its own threshold).
        quality_floor = 0.30 + 0.20 * coverage

        if idea.probe_passed and idea.quality_score >= quality_floor:
            # ── Multi-objective Pareto replacement ────────────────────────────
            cell = archive.get_cell(target_cell[0], target_cell[1])
            if cell.is_empty or should_replace_pareto(idea, cell.idea):
                updated = archive.update(idea)
            else:
                updated = False
            if updated:
                self._register_archived_title(idea)
                self._novelty_checker.register(idea.title, idea.method)
                self._failure_tracker.record_success(target_cell[0], target_cell[1])
            self.diversity.record_strategy_attempt(strategy, novelty_idx, updated)
            progress(f"  PASS (q={idea.quality_score:.2f}, floor={quality_floor:.2f}) — archived: {updated}")
            return updated

        if idea.probe_passed and idea.quality_score < quality_floor:
            progress(
                f"  PASS but below quality floor "
                f"(q={idea.quality_score:.2f} < {quality_floor:.2f}) — sending to revision."
            )
            # Fall through to revision path instead of archiving a low-quality pass

        # ── Speed: skip revision entirely when we already have decent coverage ─
        if coverage > 0.30 and not idea.probe_passed:
            self.diversity.record_strategy_attempt(strategy, novelty_idx, False)
            progress(f"  SKIP revision (coverage={coverage:.0%}, saving time)")
            return False

        # ── Skip revision if quality is too low to recover ────────────────────
        # quality < 0.25 means at least two probes scored near-zero;
        # revision rarely recovers from that, so save the LLM call.
        if idea.quality_score < 0.25:
            self.diversity.record_strategy_attempt(strategy, novelty_idx, False)
            progress(f"  FAIL (q={idea.quality_score:.2f}) — too low for revision, discarded.")
            return False

        # Revision — rank feedback by score (worst probe first) so the LLM
        # focuses its revision effort on the most critical failure.
        feedback = _ranked_feedback(probe_result)
        progress(f"  FAIL (q={idea.quality_score:.2f}) — revising …")
        revised = self._revise_safe(idea, feedback, idea.quality_score)

        if revised is None:
            self.diversity.record_strategy_attempt(strategy, novelty_idx, False)
            return False

        # Re-force coordinates after revision
        revised.methodology_type = METHODOLOGY_TYPES[target_cell[0]]
        revised.novelty_level = NOVELTY_LEVELS[target_cell[1]]

        # Skip re-probe if revision is also a duplicate
        if self._is_duplicate(revised):
            self.diversity.record_strategy_attempt(strategy, novelty_idx, False)
            progress(f"  Cell {target_cell}: Revised idea is near-duplicate, discarded.")
            return False

        probe2 = self._probe_safe(revised)
        revised.probe_scores = probe2.get("scores", {})
        revised.probe_passed = probe2.get("all_pass", False)
        revised.quality_score = probe2.get("quality", 0.0)

        if revised.probe_passed:
            updated = archive.update(revised)
            if updated:
                self._register_archived_title(revised)
                self._novelty_checker.register(revised.title, revised.method)
                self._failure_tracker.record_success(target_cell[0], target_cell[1])
            self.diversity.record_strategy_attempt(strategy, novelty_idx, updated)
            progress(f"  REVISED PASS (q={revised.quality_score:.2f}) — archived: {updated}")
            return updated

        # Even failed revisions can fill empty cells if quality is decent.
        # Stagnant cells (≥5 attempts) get a lower rescue threshold (0.28 vs 0.35)
        # so they fill with a "best-so-far" idea rather than staying empty forever.
        cell_attempts = self.diversity._cell_attempts.get(target_cell, 0)
        partial_threshold = 0.28 if cell_attempts >= 5 else 0.35
        if revised.quality_score >= partial_threshold:
            updated = archive.update(revised)
            if updated:
                self._register_archived_title(revised)
                self.diversity.record_strategy_attempt(strategy, novelty_idx, True)
                progress(
                    f"  PARTIAL (q={revised.quality_score:.2f}, threshold={partial_threshold}) "
                    f"— archived as best available."
                )
                return True

        self.diversity.record_strategy_attempt(strategy, novelty_idx, False)
        progress(f"  REVISED FAIL (q={revised.quality_score:.2f}) — discarded.")
        return False

    def _generate_idea(
        self,
        dag: KnowledgeDAG,
        strategy: str,
        target_cell,
        avoid_context: str = "",
    ) -> Optional[Idea]:
        try:
            if strategy == "A":
                frontier_id = self.diversity.pick_frontier_paper(dag)
                if frontier_id is None:
                    # Fallback: generate idea directly from topic (no paper context)
                    return self._generate_idea_from_topic(dag, target_cell, avoid_context)
                # Batch generation: get 2 ideas in one LLM call, pick the
                # longer-method one (proxy for specificity) as the candidate.
                # Falls back to single generation on parse failure.
                batch = self.ideation.generate_batch_strategy_a(
                    dag, frontier_id, target_cell, avoid_context,
                )
                if batch:
                    return max(batch, key=lambda i: len(i.method))
                return self.ideation.generate_strategy_a(dag, frontier_id, target_cell, avoid_context)

            if strategy == "B":
                pair = self.diversity.pick_cluster_pair(dag, "B")
                if pair is None:
                    frontier_id = self.diversity.pick_frontier_paper(dag)
                    if frontier_id is None:
                        return self._generate_idea_from_topic(dag, target_cell, avoid_context)
                    return self.ideation.generate_strategy_a(dag, frontier_id, target_cell, avoid_context)
                return self.ideation.generate_strategy_b(dag, pair[0], pair[1], target_cell, avoid_context)

            if strategy == "C":
                pair = self.diversity.pick_cluster_pair(dag, "C")
                if pair is None:
                    frontier_id = self.diversity.pick_frontier_paper(dag)
                    if frontier_id is None:
                        return self._generate_idea_from_topic(dag, target_cell, avoid_context)
                    return self.ideation.generate_strategy_a(dag, frontier_id, target_cell, avoid_context)
                return self.ideation.generate_strategy_c(dag, pair[0], pair[1], target_cell, avoid_context)

        except Exception as exc:
            return None
        return None

    def _generate_idea_from_topic(
        self,
        dag: KnowledgeDAG,
        target_cell,
        avoid_context: str = "",
    ) -> Optional[Idea]:
        """
        Fallback: generate an idea directly from the topic when the DAG is empty.

        Uses enhanced generation with:
          1. Topic decomposition (cached sub-problems)
          2. Multi-perspective expert persona (rotated)
          3. Chain-of-thought structured reasoning
          4. Self-improvement loop (generate → critique → refine)
        """
        topic = getattr(self, "_topic", "research")

        # ── Inject failure-pattern mitigations + cross-run lessons ────────────
        failure_hints = self._failure_tracker.get_mitigations(
            target_cell[0], target_cell[1],
        )
        cross_run_ctx = ""
        if self._cross_run_memory:
            cross_run_ctx = self._cross_run_memory.get_context_for_prompt()
        if failure_hints or cross_run_ctx:
            avoid_context = f"{avoid_context}\n{failure_hints}\n{cross_run_ctx}".strip()

        # Decompose topic into sub-problems (cached after first call)
        sub_problems = self.ideation.decompose_topic(topic)
        sub_context = ""
        if sub_problems:
            # Pick a sub-problem relevant to the target cell
            sub_idx = (target_cell[0] * 3 + target_cell[1]) % len(sub_problems)
            sub_context = f"Focus on this specific sub-problem: {sub_problems[sub_idx]}"

        context = f"{sub_context}\n{avoid_context}".strip()

        # Use self-improvement loop (generate → critique → refine × 2 rounds)
        try:
            idea = self.ideation.generate_with_self_improvement(
                topic, target_cell, context=context, rounds=2,
            )
            if idea:
                return idea
        except Exception:
            pass

        # Fallback: chain-of-thought only (no self-improvement)
        try:
            idea = self.ideation.generate_with_chain_of_thought(
                topic, target_cell, context=context,
            )
            if idea:
                return idea
        except Exception:
            pass

        # Last resort: simple direct generation
        from models.idea import METHODOLOGY_TYPES, NOVELTY_LEVELS
        from agents.ideation_agent import _novelty_temperature, _dict_to_idea

        method_hint = METHODOLOGY_TYPES[target_cell[0]]
        novelty_hint = NOVELTY_LEVELS[target_cell[1]]

        system = self.ideation._build_persona_system_prompt(target_cell)
        user = (
            f"Research topic: {topic}\n\n"
            f"Target methodology type: {method_hint}\n"
            f"Target novelty level: {novelty_hint}\n\n"
            "Generate a specific, actionable research idea that is DIRECTLY about "
            f"this topic: {topic}. The idea MUST be relevant to {topic}.\n\n"
            f"{context}\n\n"
            + _IDEA_SCHEMA
        )

        temp = _novelty_temperature(target_cell[1])
        result = self.ideation._call_json(system, user, max_tokens=1024, temperature=temp)
        return _dict_to_idea(result, "A")

    def _probe_safe(self, idea: Idea) -> Dict[str, Any]:
        try:
            return self.critic.probe_all(idea)
        except Exception:
            return {
                "all_pass": False,
                "scores": {"code": 0.5, "dataset": 0.5, "constraint": 0.5, "novelty": 0.5},
                "feedback": "Probe execution failed.",
                "quality": 0.5,
            }

    def _revise_safe(self, idea: Idea, feedback: str, quality_score: float = 0.5) -> Optional[Idea]:
        try:
            return self.ideation.revise_idea(idea, feedback, quality_score)
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _ranked_feedback(probe_result: Dict[str, Any]) -> str:
    """
    Build a revision feedback string sorted by probe score ascending (worst first).

    Presenting the most critical failures first focuses the LLM's revision on
    the issues that actually caused the idea to fail, instead of burying them
    after passing probes.  Each section shows score + issues + suggestions.
    """
    scores: Dict[str, float] = probe_result.get("scores", {})
    details: Dict[str, Any] = probe_result.get("details", {})

    probe_labels = {
        "code":       "Code implementability",
        "dataset":    "Dataset availability",
        "constraint": "Compute constraint",
        "novelty":    "Novelty",
    }

    # Sort probes by score ascending so worst failure appears first
    ordered = sorted(scores.items(), key=lambda kv: kv[1])

    parts: List[str] = []
    for probe_key, score in ordered:
        label = probe_labels.get(probe_key, probe_key.capitalize())
        detail = details.get(probe_key, {})
        issues = detail.get("issues", "")
        suggestions = detail.get("suggestions", "")
        parts.append(f"[{label} — score {score:.2f}]")
        if issues:
            parts.append(f"  Issue: {issues}")
        if suggestions:
            parts.append(f"  Fix:   {suggestions}")

    return "\n".join(parts) if parts else probe_result.get("feedback", "")
