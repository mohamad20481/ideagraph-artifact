"""
agents/knowledge_architect.py - Builds the literature Knowledge DAG.

Pipeline:
  1. Generate seed queries  (LLM)
  2. Retrieve & score papers (Semantic Scholar + heuristic ranking)
  3. BFS expansion          (forward / backward / lateral)
  4. Classify edges         (LLM, batched)
  5. Detect communities     (python-louvain)
  6. Annotate clusters      (LLM)
  7. Identify frontiers
"""

from __future__ import annotations
import json
import re
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import config
from models.dag import KnowledgeDAG, Paper, EDGE_TYPES
from tools.semantic_scholar import search_papers, get_citations, get_references
from .base_agent import BaseAgent

try:
    import community as community_louvain  # python-louvain
    import networkx as nx
    _LOUVAIN_OK = True
except ImportError:
    _LOUVAIN_OK = False
    import networkx as nx


class KnowledgeArchitect(BaseAgent):
    """Builds a typed, clustered KnowledgeDAG from a research topic string."""

    def __init__(self) -> None:
        super().__init__(temperature=0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────
    def build_dag(
        self,
        topic: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> KnowledgeDAG:
        dag = KnowledgeDAG()

        def _progress(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        # 1. Seed queries
        _progress("Generating seed queries …")
        queries = self._generate_seed_queries(topic)

        # 2. Retrieve papers — parallel queries with thread pool
        #    Semantic Scholar rate-limiter in _get() serialises the actual HTTP
        #    calls, so launching queries concurrently just fills the pipeline
        #    and removes the Python-side idle time between requests.
        _progress("Searching Semantic Scholar …")
        all_candidates: Dict[str, Dict[str, Any]] = {}

        def _search_one(q: str) -> List[Dict[str, Any]]:
            return search_papers(q, limit=20, api_key=config.SEMANTIC_SCHOLAR_API_KEY)

        with ThreadPoolExecutor(max_workers=min(len(queries), 3)) as pool:
            futures = {pool.submit(_search_one, q): (i, q) for i, q in enumerate(queries)}
            for future in as_completed(futures):
                i, q = futures[future]
                try:
                    batch = future.result(timeout=60)
                except Exception:
                    batch = []
                _progress(f"  Query {i+1}/{len(queries)}: {q[:60]}")
                for p in batch:
                    pid = p.get("paperId", "")
                    if pid and pid not in all_candidates:
                        all_candidates[pid] = p
                _progress(f"  → {len(batch)} papers found (total unique: {len(all_candidates)})")

        # 3. Score and select top SEEDS
        _progress("Scoring and selecting seed papers …")
        seeds = self._select_seeds(topic, list(all_candidates.values()), n=config.SEEDS)

        # Add seeds to DAG
        for raw in seeds:
            paper = _raw_to_paper(raw)
            dag.add_paper(paper)

        # 4. BFS expansion
        _progress("Expanding graph via BFS …")
        self._bfs_expand(dag, topic, on_progress=_progress)

        # 5. Classify edges
        _progress("Classifying edges …")
        self._classify_edges(dag, topic)

        # 6. Community detection
        _progress("Detecting communities …")
        self._detect_communities(dag)

        # 7. Annotate clusters
        _progress("Annotating clusters …")
        self._annotate_clusters(dag, topic)

        # 8. Frontiers
        for p in dag.get_frontier_nodes():
            p.is_frontier = True

        _progress("DAG construction complete.")
        return dag

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 – Seed queries
    # ─────────────────────────────────────────────────────────────────────────
    def _generate_seed_queries(self, topic: str) -> List[str]:
        system = (
            "You are a research librarian. Output ONLY a JSON array of exactly 5 search queries "
            "(strings). No explanation."
        )
        user = (
            f"Topic: {topic}\n"
            "Generate 5 diverse Semantic Scholar search queries covering:\n"
            "1. core methodology\n2. applications\n3. theoretical foundations\n"
            "4. evaluation approaches\n5. interdisciplinary connections\n"
            "Return: [\"query1\", \"query2\", ...]"
        )
        raw = self._call(system, user, max_tokens=512, temperature=0.0)
        try:
            match = re.search(r"\[[\s\S]*?\]", raw)
            if match:
                queries = json.loads(match.group(0))
                if isinstance(queries, list):
                    return [str(q) for q in queries[:config.SEED_QUERIES]]
        except Exception:
            pass
        # Fallback: generate simple queries
        words = topic.split()
        return [
            topic,
            f"{topic} survey",
            f"{topic} applications",
            f"{topic} theoretical foundations",
            " ".join(words[:3]) + " evaluation" if len(words) >= 3 else topic + " benchmark",
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 – Paper scoring
    # ─────────────────────────────────────────────────────────────────────────
    _CURRENT_YEAR = 2026  # update if reusing in a later year

    @staticmethod
    def _citation_velocity(raw: Dict[str, Any]) -> float:
        """
        Citations per year since publication.

        Velocity captures impact-per-time better than raw citation count:
        - A 2023 paper with 50 citations is more impactful than a
          2010 paper with 100 citations.
        - Papers from the current year get at least 1 year of credit.
        """
        citations = int(raw.get("citationCount") or 0)
        year = int(raw.get("year") or 0)
        years_active = max(1, KnowledgeArchitect._CURRENT_YEAR - year)
        return citations / years_active

    def _score_paper(
        self,
        raw: Dict[str, Any],
        topic: str,
        velocity_rank: float,
        recency_rank: float,
    ) -> float:
        """
        Heuristic score: 0.4*velocity_rank + 0.3*recency_rank + 0.3*sim_rank.

        velocity_rank replaces the old raw citation_rank — it gives normalised
        position in the per-candidate velocity sort, so recent impactful papers
        score higher than old papers with merely large citation counts.
        Uses _paper_text_tokens() cache — same paper scored across multiple
        seed query result sets pays tokenisation cost only once.
        """
        topic_words = self._topic_words(topic)
        sim_rank = len(topic_words & self._paper_text_tokens(raw)) / len(topic_words) if topic_words else 0.0
        return 0.4 * velocity_rank + 0.3 * recency_rank + 0.3 * sim_rank

    def _select_seeds(
        self,
        topic: str,
        candidates: List[Dict[str, Any]],
        n: int = 10,
        mmr_lambda: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        MMR (Maximum Marginal Relevance) seed selection.

        Pure top-N scoring clusters seeds in the same subfield.  MMR alternates
        between relevance and diversity to spread seeds across the topic space:

            MMR_score = λ · relevance_score  −  (1−λ) · max_sim_to_selected

        λ=0.6 balances quality (60%) and diversity (40%).  Papers that are
        highly similar to already-selected seeds are penalised, so the final
        set covers more of the literature.

        Also pre-filters candidates with abstracts < 50 chars (same guard as BFS).
        """
        if not candidates:
            return []

        # ── Abstract quality pre-filter ───────────────────────────────────────
        candidates = [
            p for p in candidates
            if len((p.get("abstract") or "").strip()) >= 50
        ]
        if not candidates:
            return []

        # ── Compute normalised relevance scores ───────────────────────────────
        sorted_by_velocity = sorted(candidates, key=self._citation_velocity)
        sorted_by_year = sorted(candidates, key=lambda p: p.get("year") or 0)
        N = len(candidates)

        rank_velocity = {p["paperId"]: i / max(N - 1, 1) for i, p in enumerate(sorted_by_velocity)}
        rank_year = {p["paperId"]: i / max(N - 1, 1) for i, p in enumerate(sorted_by_year)}

        rel_scores: Dict[str, float] = {}
        for p in candidates:
            pid = p.get("paperId", "")
            if not pid:
                continue
            rel_scores[pid] = self._score_paper(
                p, topic, rank_velocity.get(pid, 0.0), rank_year.get(pid, 0.0)
            )

        valid = [p for p in candidates if p.get("paperId") and p["paperId"] in rel_scores]
        if not valid:
            return []

        # ── MMR iterative selection ───────────────────────────────────────────
        # Precompute token sets once (uses existing cache)
        tok: Dict[str, frozenset] = {
            p["paperId"]: self._paper_text_tokens(p) for p in valid
        }

        selected: List[Dict[str, Any]] = []
        remaining = list(valid)

        # First pick: pure highest-relevance (no selected set yet)
        remaining.sort(key=lambda p: rel_scores.get(p["paperId"], 0.0), reverse=True)
        selected.append(remaining.pop(0))

        while len(selected) < n and remaining:
            sel_toks = [tok[p["paperId"]] for p in selected]
            best_score = float("-inf")
            best_idx = 0

            for idx, p in enumerate(remaining):
                pid = p["paperId"]
                pt = tok[pid]
                # Max Jaccard similarity to any already-selected paper
                max_sim = 0.0
                for st in sel_toks:
                    union = pt | st
                    if union:
                        max_sim = max(max_sim, len(pt & st) / len(union))

                mmr = mmr_lambda * rel_scores[pid] - (1.0 - mmr_lambda) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx

            selected.append(remaining.pop(best_idx))

        return selected

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 – BFS expansion (parallel batch processing)
    # ─────────────────────────────────────────────────────────────────────────
    _BFS_BATCH = 4  # process up to N nodes simultaneously per BFS round

    def _bfs_expand(
        self,
        dag: KnowledgeDAG,
        topic: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        BFS over the paper graph.  Each round collects up to _BFS_BATCH nodes
        from the front of the queue and expands them in parallel.  Each node
        fires 3 parallel Semantic Scholar requests + 1 LLM call, so batching
        multiple nodes gives a large wall-clock speedup.
        """
        visited: Set[str] = set(dag._papers.keys())
        visited_lock = threading.Lock()          # guards visited & dag size check
        queue: deque[Tuple[str, int]] = deque(
            [(pid, 0) for pid in list(dag._papers.keys())]
        )
        seed_ids = frozenset(dag._papers.keys())

        # Persistent pool for the entire BFS traversal.  Sized to handle
        # up to _BFS_BATCH node expansions × 3 API calls each, plus the
        # outer batch-dispatch work.  Previously a new ThreadPoolExecutor
        # was created per BFS round (outer) AND per node (inner), spawning
        # dozens of throwaway pool-manager threads across a typical run.
        _bfs_pool = ThreadPoolExecutor(max_workers=self._BFS_BATCH * 3 + 2)

        try:
            while queue and len(dag._papers) < config.MAX_NODES:
                # ── Collect a batch of ready nodes ────────────────────────
                batch: List[Tuple[str, int]] = []
                while queue and len(batch) < self._BFS_BATCH:
                    current_id, depth = queue.popleft()
                    if depth >= config.DEPTH:
                        continue
                    if dag.get_paper(current_id) is None:
                        continue
                    batch.append((current_id, depth))

                if not batch:
                    continue

                # ── Expand all nodes in the batch in parallel ─────────────
                new_queue_entries: List[Tuple[str, int]] = []
                if len(batch) == 1:
                    new_queue_entries = self._expand_one_node(
                        dag, batch[0][0], batch[0][1],
                        visited, visited_lock, seed_ids, topic,
                        pool=_bfs_pool,
                    )
                else:
                    futures = {
                        _bfs_pool.submit(
                            self._expand_one_node,
                            dag, nid, depth, visited, visited_lock, seed_ids, topic,
                            _bfs_pool,
                        ): (nid, depth)
                        for nid, depth in batch
                    }
                    for future in as_completed(futures, timeout=120):
                        try:
                            new_queue_entries.extend(future.result(timeout=120))
                        except Exception:
                            pass

                for pid, next_depth in new_queue_entries:
                    queue.append((pid, next_depth))
        finally:
            _bfs_pool.shutdown(wait=False)

    def _expand_one_node(
        self,
        dag: KnowledgeDAG,
        current_id: str,
        depth: int,
        visited: Set[str],
        visited_lock: "threading.Lock",
        seed_ids: frozenset,
        topic: str,
        pool: Optional[ThreadPoolExecutor] = None,
    ) -> List[Tuple[str, int]]:
        """
        Expand a single BFS node.  Thread-safe: uses visited_lock for all
        check-and-claim operations so parallel nodes never add the same paper.
        Returns list of (new_paper_id, next_depth) pairs to enqueue.
        """
        current_paper = dag.get_paper(current_id)
        if current_paper is None:
            return []

        # ── Fetch forward, backward, lateral in parallel ──────────────────
        other_seed = next((s for s in seed_ids if s != current_id), None) if len(seed_ids) > 1 else None

        def _fetch_forward():
            return get_citations(current_id, limit=config.FORWARD_BRANCH * 5, api_key=config.SEMANTIC_SCHOLAR_API_KEY)

        def _fetch_backward():
            return get_references(current_id, limit=config.BACKWARD_BRANCH * 4, api_key=config.SEMANTIC_SCHOLAR_API_KEY)

        def _fetch_lateral():
            if other_seed is None:
                return []
            return get_citations(other_seed, limit=20, api_key=config.SEMANTIC_SCHOLAR_API_KEY)

        fw_papers, bw_papers, lateral_raw = [], [], []
        # Reuse the shared BFS pool when available (avoids creating a
        # throwaway ThreadPoolExecutor per node expansion).
        _pool = pool or ThreadPoolExecutor(max_workers=3)
        try:
            f_fw = _pool.submit(_fetch_forward)
            f_bw = _pool.submit(_fetch_backward)
            f_lt = _pool.submit(_fetch_lateral)
            try:
                fw_papers = f_fw.result(timeout=30)
            except Exception:
                pass
            try:
                bw_papers = f_bw.result(timeout=30)
            except Exception:
                pass
            try:
                lateral_raw = f_lt.result(timeout=30)
            except Exception:
                pass
        finally:
            # Only shut down if we created a local pool (no shared pool given)
            if pool is None:
                _pool.shutdown(wait=False)

        new_entries: List[Tuple[str, int]] = []
        next_depth = depth + 1

        def _try_claim(pid: str) -> bool:
            """Atomically claim a paperId. Returns True if newly claimed."""
            with visited_lock:
                if pid in visited or len(dag._papers) >= config.MAX_NODES:
                    return False
                visited.add(pid)
                return True

        # --- Forward: papers citing current ---
        for raw in self._rank_by_relevance(fw_papers, topic)[:config.FORWARD_BRANCH]:
            pid = raw.get("paperId", "")
            if pid and raw.get("abstract") and _try_claim(pid):  # skip empty abstracts
                dag.add_paper(_raw_to_paper(raw))
                dag.add_edge(current_id, pid, "extends", "forward citation")
                new_entries.append((pid, next_depth))

        # --- Backward: references of current (heuristic/LLM selects) ---
        selected = self._llm_select_backward(current_paper, bw_papers, config.BACKWARD_BRANCH)
        for raw in selected:
            pid = raw.get("paperId", "")
            if pid and raw.get("abstract") and _try_claim(pid):  # skip empty abstracts
                dag.add_paper(_raw_to_paper(raw))
                dag.add_edge(pid, current_id, "enables", "backward reference")
                new_entries.append((pid, next_depth))

        # --- Lateral: papers citing a seed AND related to current ---
        lateral_candidates = [p for p in lateral_raw if p.get("paperId") and p.get("abstract")]
        for raw in self._rank_by_relevance(lateral_candidates, topic)[:config.LATERAL_BRANCH]:
            pid = raw.get("paperId", "")
            if pid and _try_claim(pid):
                dag.add_paper(_raw_to_paper(raw))
                dag.add_edge(current_id, pid, "combines", "lateral connection")
                new_entries.append((pid, next_depth))

        return new_entries

    def _topic_words(self, topic: str) -> set:
        """Precomputed normalised topic word set (called once per BFS node)."""
        if not hasattr(self, "_topic_words_cache"):
            self._topic_words_cache: Dict[str, set] = {}
        if topic not in self._topic_words_cache:
            self._topic_words_cache[topic] = set(
                re.sub(r"[^a-z0-9 ]", "", topic.lower()).split()
            )
        return self._topic_words_cache[topic]

    def _paper_text_tokens(self, raw: Dict) -> frozenset:
        """
        Cached frozenset of lowercase words from a paper's title + abstract.
        Papers from Semantic Scholar are immutable, so the cache never needs
        invalidation.  Benign race on first access: both threads compute the
        same value, last writer wins — correctness is preserved.
        """
        if not hasattr(self, "_paper_token_cache"):
            self._paper_token_cache: Dict[str, frozenset] = {}
        pid = raw.get("paperId", "")
        if pid and pid in self._paper_token_cache:
            return self._paper_token_cache[pid]
        text = ((raw.get("title") or "") + " " + (raw.get("abstract") or "")).lower()
        tokens = frozenset(text.split())
        if pid:
            self._paper_token_cache[pid] = tokens
        return tokens

    def _rank_by_relevance(self, papers: List[Dict], topic: str) -> List[Dict]:
        """Sort papers by title/abstract word overlap with topic (descending).
        Uses _paper_text_tokens() to avoid re-tokenizing the same papers across
        multiple BFS node expansions (common for highly-cited papers).
        """
        topic_words = self._topic_words(topic)
        if not topic_words:
            return papers

        n = len(topic_words)

        def sim(p: Dict) -> float:
            return len(topic_words.intersection(self._paper_text_tokens(p))) / n

        return sorted(papers, key=sim, reverse=True)

    def _llm_select_backward(
        self,
        current: Paper,
        references: List[Dict],
        n: int,
    ) -> List[Dict]:
        """Use LLM to pick the n most foundational references for current paper.
        Skips the LLM call when the candidate list is small (≤ 2×n) — just
        use heuristic ranking, saving an entire round-trip per BFS node.
        """
        if not references:
            return []
        if len(references) <= n * 2:
            # Not worth an LLM call — rank by citation velocity (impact/year)
            return sorted(
                references,
                key=self._citation_velocity,
                reverse=True,
            )[:n]

        ref_list = "\n".join(
            f"{i}. {r.get('title', 'Untitled')} ({r.get('year', '?')})"
            for i, r in enumerate(references[:20])
        )
        system = "You are a research expert. Output ONLY a JSON array of integer indices."
        user = (
            f"Paper: \"{current.title}\"\n"
            f"Select the {n} most foundational references (by index, 0-based):\n{ref_list}\n"
            f"Return: [idx1, idx2, ...]"
        )
        raw = self._call(system, user, max_tokens=128, temperature=0.0)
        try:
            match = re.search(r"\[[\s\S]*?\]", raw)
            if match:
                indices = json.loads(match.group(0))
                valid = [references[i] for i in indices if isinstance(i, int) and 0 <= i < len(references)]
                return valid[:n]
        except Exception:
            pass
        return references[:n]

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 – Edge classification (batched)
    # ─────────────────────────────────────────────────────────────────────────
    def _classify_edges(self, dag: KnowledgeDAG, topic: str) -> None:
        edges = list(dag.graph.edges(data=True))
        if not edges:
            return

        # Batch into groups of 10 and classify all batches in parallel
        batches = [edges[i: i + 10] for i in range(0, len(edges), 10)]
        with ThreadPoolExecutor(max_workers=min(len(batches), 4)) as pool:
            futures = [pool.submit(self._classify_edge_batch, dag, batch) for batch in batches]
            for future in as_completed(futures, timeout=180):
                try:
                    future.result(timeout=180)
                except Exception:
                    pass

    def _classify_edge_batch(self, dag: KnowledgeDAG, batch: List[Tuple]) -> None:
        edge_descs = []
        for i, (u, v, _) in enumerate(batch):
            pu = dag.get_paper(u)
            pv = dag.get_paper(v)
            if pu is None or pv is None:
                continue
            edge_descs.append(
                f"{i}. FROM \"{pu.title}\" TO \"{pv.title}\""
            )

        if not edge_descs:
            return

        types_str = ", ".join(EDGE_TYPES)
        system = (
            "Classify research paper relationships. "
            f"Edge types: {types_str}. "
            "Output ONLY a JSON object mapping index (string) to edge type string."
        )
        user = "\n".join(edge_descs) + f"\nReturn: {{\"0\": \"type\", \"1\": \"type\", ...}}"
        raw = self._call(system, user, max_tokens=512, temperature=0.0)

        try:
            match = re.search(r"\{[\s\S]*?\}", raw)
            if match:
                classifications = json.loads(match.group(0))
                for i, (u, v, data) in enumerate(batch):
                    edge_type = classifications.get(str(i), "extends")
                    if edge_type not in EDGE_TYPES:
                        edge_type = "extends"
                    dag.set_edge_type(u, v, edge_type)  # thread-safe write
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5 – Community detection
    # ─────────────────────────────────────────────────────────────────────────
    def _detect_communities(self, dag: KnowledgeDAG) -> None:
        if len(dag._papers) < 2:
            for pid, paper in dag._papers.items():
                paper.cluster_id = 0
            return

        undirected = dag.graph.to_undirected()

        if _LOUVAIN_OK:
            try:
                partition = community_louvain.best_partition(undirected)
                for pid, cid in partition.items():
                    paper = dag.get_paper(pid)
                    if paper:
                        paper.cluster_id = cid
                        dag.graph.nodes[pid]["cluster_id"] = cid
                return
            except Exception:
                pass

        # Fallback: connected components
        for i, component in enumerate(nx.connected_components(undirected)):
            for pid in component:
                paper = dag.get_paper(pid)
                if paper:
                    paper.cluster_id = i
                    dag.graph.nodes[pid]["cluster_id"] = i

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6 – Cluster annotation
    # ─────────────────────────────────────────────────────────────────────────
    def _annotate_clusters(self, dag: KnowledgeDAG, topic: str) -> None:
        cluster_ids = dag.get_cluster_ids()

        def _annotate_one(cid: int) -> Tuple[int, Dict[str, Any]]:
            papers = dag.get_papers_in_cluster(cid)
            if not papers:
                return cid, {"theme": f"Cluster {cid}", "open_questions": [], "maturity": "developing"}

            titles = "; ".join(p.title for p in papers[:8])
            system = (
                "You are a research analyst. Output ONLY valid JSON with keys: "
                "theme (string), open_questions (list of 3-5 strings), maturity (one of: emerging, developing, mature)."
            )
            user = (
                f"Research topic: {topic}\n"
                f"Cluster {cid} papers: {titles}\n"
                "Summarise this research cluster."
            )
            result = self._call_json(system, user, max_tokens=512, temperature=0.0)
            if result:
                return cid, {
                    "theme": result.get("theme", f"Cluster {cid}"),
                    "open_questions": result.get("open_questions", []),
                    "maturity": result.get("maturity", "developing"),
                }
            return cid, {"theme": f"Research cluster {cid}", "open_questions": [], "maturity": "developing"}

        # Annotate all clusters in parallel
        with ThreadPoolExecutor(max_workers=min(len(cluster_ids), 6)) as pool:
            futures = {pool.submit(_annotate_one, cid): cid for cid in cluster_ids}
            for future in as_completed(futures):
                try:
                    cid, meta = future.result()
                    dag.cluster_metadata[cid] = meta
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────
def _raw_to_paper(raw: Dict[str, Any]) -> Paper:
    return Paper(
        paper_id=raw.get("paperId", ""),
        title=raw.get("title") or "Untitled",
        abstract=raw.get("abstract") or "",
        year=int(raw.get("year") or 0),
        citation_count=int(raw.get("citationCount") or 0),
    )
