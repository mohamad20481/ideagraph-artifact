"""
models/dag.py - Knowledge DAG built from literature papers.

Nodes  : Paper objects (stored as node attributes)
Edges  : directed, typed (one of EDGE_TYPES)
"""

from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import networkx as nx


# ── Semantic edge vocabulary ──────────────────────────────────────────────────
EDGE_TYPES: List[str] = [
    "extends",      # builds directly on the source paper
    "applies",      # applies source methods to a new domain
    "combines",     # merges ideas from source with another stream
    "contrasts",    # takes a different or opposing stance
    "generalizes",  # subsumes source as a special case
    "enables",      # provides a foundational tool/dataset/theory
]


# ── Paper data model ──────────────────────────────────────────────────────────
# slots=True: per-instance memory drops ~40-50%. The DAG can hold hundreds
# of Paper objects from Semantic Scholar lookups; this matters for memory.
@dataclass(slots=True)
class Paper:
    paper_id: str
    title: str
    abstract: str
    year: int
    citation_count: int
    cluster_id: Optional[int] = None
    is_frontier: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract[:300] if self.abstract else "",
            "year": self.year,
            "citation_count": self.citation_count,
            "cluster_id": self.cluster_id,
            "is_frontier": self.is_frontier,
        }


# ── KnowledgeDAG ─────────────────────────────────────────────────────────────
class KnowledgeDAG:
    """
    Directed acyclic knowledge graph over research papers.
    Clusters are detected externally and stored as metadata here.
    Thread-safe: all mutations acquire _lock.
    """

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self._papers: Dict[str, Paper] = {}
        self.cluster_metadata: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._frontier_cache: Optional[List[Paper]] = None  # invalidated on mutation
        self._cluster_paper_cache: Dict[int, List[Paper]] = {}  # invalidated on mutation

    # ── Mutation helpers ──────────────────────────────────────────────────────
    def add_paper(self, paper: Paper) -> None:
        """Add or update a paper node (thread-safe)."""
        with self._lock:
            self._papers[paper.paper_id] = paper
            self.graph.add_node(paper.paper_id, **paper.to_dict())
            self._frontier_cache = None  # graph changed — invalidate
            self._cluster_paper_cache.clear()

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        edge_type: str,
        reasoning: str = "",
    ) -> None:
        """Add a directed typed edge (thread-safe)."""
        with self._lock:
            if from_id not in self._papers or to_id not in self._papers:
                return
            if from_id == to_id:
                return
            if edge_type not in EDGE_TYPES:
                edge_type = "extends"
            self.graph.add_edge(from_id, to_id, edge_type=edge_type, reasoning=reasoning)
            self._frontier_cache = None  # out-degrees changed — invalidate

    def set_edge_type(self, from_id: str, to_id: str, edge_type: str) -> None:
        """Update edge type attribute (thread-safe)."""
        with self._lock:
            if self.graph.has_edge(from_id, to_id):
                self.graph[from_id][to_id]["edge_type"] = edge_type

    # ── Query helpers ─────────────────────────────────────────────────────────
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        return self._papers.get(paper_id)

    def get_all_papers(self) -> List[Paper]:
        return list(self._papers.values())

    def get_frontier_nodes(self) -> List[Paper]:
        """Papers that have no outgoing edges (research frontier).
        Result is cached after DAG construction and reused during ideation
        (the DAG is static once build_dag() returns, so the cache stays valid).
        The cache is invalidated by add_paper() and add_edge().
        """
        if self._frontier_cache is not None:
            return self._frontier_cache
        frontier = []
        for pid, paper in self._papers.items():
            if pid in self.graph and self.graph.out_degree(pid) == 0:
                paper.is_frontier = True
                frontier.append(paper)
        self._frontier_cache = frontier
        return frontier

    def get_papers_in_cluster(self, cluster_id: int) -> List[Paper]:
        """Cached cluster → papers lookup.  O(1) after first call for each cluster.
        Cache is invalidated when papers are added (cluster assignments may change).
        """
        if cluster_id in self._cluster_paper_cache:
            return self._cluster_paper_cache[cluster_id]
        result = [p for p in self._papers.values() if p.cluster_id == cluster_id]
        self._cluster_paper_cache[cluster_id] = result
        return result

    def get_cluster_ids(self) -> List[int]:
        ids = set(p.cluster_id for p in self._papers.values() if p.cluster_id is not None)
        return sorted(ids)

    def get_cluster_paper_ids(self, cluster_id: int) -> Set[str]:
        """Return set of paper_ids for a cluster (fast lookup)."""
        return {p.paper_id for p in self._papers.values() if p.cluster_id == cluster_id}

    def has_cross_cluster_edge(self, c1: int, c2: int) -> bool:
        """
        Return True if at least one edge exists between papers of c1 and c2.
        O(|c1|*|c2|) worst case but uses set intersection on neighbour lists
        which is much faster than scanning all edges.
        """
        ids_c1 = self.get_cluster_paper_ids(c1)
        ids_c2 = self.get_cluster_paper_ids(c2)
        for pid in ids_c1:
            # Check if any neighbour (in or out) belongs to c2
            if ids_c2.intersection(self.graph.successors(pid)):
                return True
            if ids_c2.intersection(self.graph.predecessors(pid)):
                return True
        return False

    # ── Serialisation ─────────────────────────────────────────────────────────
    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Compact representation suitable for stuffing into LLM prompts.
        Keeps only the most relevant fields.
        """
        papers_list = []
        for p in self._papers.values():
            papers_list.append({
                "id": p.paper_id,
                "title": p.title,
                "year": p.year,
                "cluster": p.cluster_id,
                "frontier": p.is_frontier,
                "citations": p.citation_count,
            })

        edges_list = []
        for u, v, data in self.graph.edges(data=True):
            edges_list.append({
                "from": u,
                "to": v,
                "type": data.get("edge_type", "extends"),
            })

        clusters_list = []
        for cid, meta in self.cluster_metadata.items():
            clusters_list.append({
                "cluster_id": cid,
                "theme": meta.get("theme", ""),
                "open_questions": meta.get("open_questions", []),
                "maturity": meta.get("maturity", ""),
                "paper_count": len(self.get_papers_in_cluster(cid)),
            })

        return {
            "node_count": len(self._papers),
            "edge_count": self.graph.number_of_edges(),
            "papers": papers_list,
            "edges": edges_list,
            "clusters": clusters_list,
        }
