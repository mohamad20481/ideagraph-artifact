"""Tests for pipeline_simulator.py — admin-facing IdeaGraph walkthrough."""
from __future__ import annotations

import pytest

from pipeline_simulator import (
    PIPELINE_STAGES,
    METHODOLOGY_TYPES,
    NOVELTY_LEVELS,
    FakeIdea,
    FakeRun,
    FakeDAGNode,
    FakeDAGEdge,
    fake_pipeline_run,
    architecture_figure,
    archive_heatmap_figure,
    funnel_figure,
    dag_figure,
)


class TestStageCatalog:
    def test_eight_stages_defined(self):
        # Knowledge → Diversity → Generation → Dedup → Probe → Revision → Archive → Debate
        assert len(PIPELINE_STAGES) == 8

    def test_each_stage_has_required_fields(self):
        required = {"icon", "name", "agent", "purpose",
                    "inputs", "outputs", "color"}
        for s in PIPELINE_STAGES:
            assert required.issubset(s.keys())
            for k in required:
                assert s[k], f"empty {k} in stage {s.get('name')}"

    def test_stage_names_are_unique(self):
        names = [s["name"] for s in PIPELINE_STAGES]
        assert len(set(names)) == len(names)

    def test_execution_revision_is_listed(self):
        # The new feature should appear in the educational walkthrough
        agents = [s["agent"] for s in PIPELINE_STAGES]
        assert any("execution_revisor" in a for a in agents)

    def test_methodology_and_novelty_match_real_models(self):
        from models.idea import METHODOLOGY_TYPES as REAL_M, NOVELTY_LEVELS as REAL_N
        assert METHODOLOGY_TYPES == REAL_M
        assert NOVELTY_LEVELS == REAL_N


class TestFakePipelineRun:
    def test_generates_requested_idea_count(self):
        run = fake_pipeline_run("topic", seed=42, n_ideas=10)
        assert isinstance(run, FakeRun)
        assert len(run.ideas) == 10

    def test_deterministic_with_same_seed(self):
        a = fake_pipeline_run("topic", seed=42, n_ideas=12)
        b = fake_pipeline_run("topic", seed=42, n_ideas=12)
        assert [i.title for i in a.ideas] == [i.title for i in b.ideas]
        assert a.dag_nodes == b.dag_nodes
        assert a.dag_edges == b.dag_edges

    def test_different_seed_yields_different_run(self):
        a = fake_pipeline_run("topic", seed=42, n_ideas=12)
        b = fake_pipeline_run("topic", seed=999, n_ideas=12)
        assert [i.title for i in a.ideas] != [i.title for i in b.ideas]

    def test_dag_size_in_reasonable_range(self):
        run = fake_pipeline_run("topic", seed=42)
        assert 30 <= run.dag_nodes <= 90
        assert 60 <= run.dag_edges <= 200

    def test_archive_respects_pareto_replacement(self):
        run = fake_pipeline_run("topic", seed=42, n_ideas=24)
        # Each archive cell holds exactly the highest-quality idea seen for it
        for (mi, ni), idea in run.archive.items():
            cell_ideas = [
                i for i in run.ideas
                if i.method_type == METHODOLOGY_TYPES[mi]
                and i.novelty == NOVELTY_LEVELS[ni]
                and (not i.rejected_reason or "replaced" in i.rejected_reason
                     or "already holds" in i.rejected_reason)
            ]
            if cell_ideas:
                qs = [
                    i.blended_quality if i.blended_quality is not None else i.quality
                    for i in cell_ideas
                ]
                idea_q = (idea.blended_quality if idea.blended_quality is not None
                            else idea.quality)
                assert idea_q == max(qs)

    def test_coverage_never_exceeds_full_grid(self):
        run = fake_pipeline_run("topic", seed=42, n_ideas=40)
        assert 0.0 <= run.coverage <= 1.0
        assert len(run.archive) <= 21

    def test_mean_quality_zero_when_archive_empty(self):
        # Pathological topic where every idea fails → archive empty
        run = FakeRun(topic="x", seed=0, dag_nodes=10, dag_edges=20)
        assert run.mean_quality == 0.0

    def test_mean_quality_computes_when_archived(self):
        run = fake_pipeline_run("topic", seed=42, n_ideas=20)
        if run.archive:
            assert 0.0 < run.mean_quality < 1.0

    def test_revision_disabled_leaves_signal_unset(self):
        run = fake_pipeline_run("topic", seed=42, n_ideas=12,
                                  enable_revision=False)
        for i in run.ideas:
            assert i.execution_signal is None
            assert i.execution_trust is None
            assert i.blended_quality is None

    def test_revision_enabled_populates_archived_ideas(self):
        run = fake_pipeline_run("topic", seed=42, n_ideas=20,
                                  enable_revision=True)
        archived = [i for i in run.ideas if i.archived]
        assert archived, "Need at least one archived idea for this test"
        for i in archived:
            assert i.execution_signal is not None
            assert i.execution_trust is not None
            assert i.blended_quality is not None
            assert 0.0 <= i.execution_signal <= 1.0
            assert 0.0 <= i.blended_quality <= 1.0

    def test_rejected_ideas_have_a_reason(self):
        run = fake_pipeline_run("topic", seed=42, n_ideas=20)
        for i in run.ideas:
            if not i.archived:
                assert i.rejected_reason, f"unarchived idea has no reason: {i.title}"

    def test_archived_ideas_are_in_archive_dict(self):
        run = fake_pipeline_run("topic", seed=42, n_ideas=24)
        archived_titles = {i.title for i in run.ideas if i.archived}
        archive_titles = {i.title for i in run.archive.values()}
        assert archived_titles == archive_titles

    def test_topic_appears_in_titles(self):
        run = fake_pipeline_run("graph neural networks", seed=42, n_ideas=8)
        # The topic keywords should appear in at least some titles
        assert any("graph neural" in i.title.lower() for i in run.ideas)


class TestFigureBuilders:
    def test_architecture_figure_returns_plotly_figure(self):
        fig = architecture_figure()
        # Either a Plotly Figure (if available) or None
        assert fig is None or hasattr(fig, "to_dict")

    def test_architecture_figure_has_one_node_per_stage(self):
        fig = architecture_figure()
        if fig is None:
            return
        # The single Scatter trace should have 8 markers
        scatter_traces = [t for t in fig.data if t.type == "scatter"]
        assert scatter_traces
        assert len(scatter_traces[0].x) == len(PIPELINE_STAGES)

    def test_archive_heatmap_uses_7x3_grid(self):
        run = fake_pipeline_run("topic", seed=42, n_ideas=24)
        fig = archive_heatmap_figure(run)
        if fig is None:
            return
        # The heatmap z-matrix should be 7 rows × 3 columns
        z = fig.data[0].z
        assert len(z) == 7
        assert all(len(row) == 3 for row in z)

    def test_funnel_figure_renders(self):
        run = fake_pipeline_run("topic", seed=42, n_ideas=20)
        fig = funnel_figure(run)
        if fig is None:
            return
        # Funnel must have all four gate stages
        labels = list(fig.data[0].y)
        assert "Generated" in labels and "Archived" in labels


class TestKnowledgeDAG:
    def test_dag_populated_after_run(self):
        run = fake_pipeline_run("topic", seed=42, n_ideas=12)
        assert len(run.dag_node_objs) == run.dag_nodes
        assert len(run.dag_edge_objs) == run.dag_edges

    def test_node_count_matches_declared(self):
        run = fake_pipeline_run("topic", seed=42, n_ideas=12)
        assert 30 <= len(run.dag_node_objs) <= 90

    def test_each_node_has_required_fields(self):
        run = fake_pipeline_run("topic", seed=42)
        for n in run.dag_node_objs:
            assert n.id and n.title
            assert 2014 <= n.year <= 2030
            assert 0 <= n.cluster
            assert n.citations >= 1

    def test_seed_papers_exist_one_per_cluster(self):
        run = fake_pipeline_run("topic", seed=42)
        seeds = [n for n in run.dag_node_objs if n.is_seed]
        clusters = {n.cluster for n in run.dag_node_objs}
        # Exactly one seed per cluster (the first node in each cluster)
        seed_clusters = {n.cluster for n in seeds}
        assert seed_clusters == clusters

    def test_frontier_nodes_are_high_citation(self):
        run = fake_pipeline_run("topic", seed=42)
        frontier = [n for n in run.dag_node_objs if n.is_frontier]
        non_frontier = [n for n in run.dag_node_objs if not n.is_frontier]
        if frontier and non_frontier:
            # Frontier nodes should have at least the citation count of the
            # max non-frontier (since they're picked from the top of the list)
            max_nonf = max(n.citations for n in non_frontier)
            min_f = min(n.citations for n in frontier)
            assert min_f >= max_nonf

    def test_edges_have_valid_endpoints(self):
        run = fake_pipeline_run("topic", seed=42)
        ids = {n.id for n in run.dag_node_objs}
        for e in run.dag_edge_objs:
            assert e.source in ids
            assert e.target in ids
            assert e.source != e.target

    def test_no_duplicate_edges(self):
        run = fake_pipeline_run("topic", seed=42)
        keys = {(e.source, e.target) for e in run.dag_edge_objs}
        # Plus reverse-key check (we treat (a,b) and (b,a) as the same edge)
        assert len(keys) == len(run.dag_edge_objs)
        for a, b in keys:
            assert (b, a) not in keys

    def test_edge_kinds_are_valid(self):
        run = fake_pipeline_run("topic", seed=42)
        for e in run.dag_edge_objs:
            assert e.kind in {"forward", "backward", "lateral"}

    def test_edge_kind_ratios_match_target(self):
        # Aggregate over multiple seeds to smooth random variation
        ratios = {"forward": [], "backward": [], "lateral": []}
        for seed in range(20):
            run = fake_pipeline_run("topic", seed=seed, n_ideas=10)
            total = len(run.dag_edge_objs)
            if total < 10:
                continue
            for kind in ratios:
                count = sum(1 for e in run.dag_edge_objs if e.kind == kind)
                ratios[kind].append(count / total)
        # Average ratios should be close to the targets ±5pp
        avg_fwd = sum(ratios["forward"]) / len(ratios["forward"])
        avg_bwd = sum(ratios["backward"]) / len(ratios["backward"])
        avg_lat = sum(ratios["lateral"]) / len(ratios["lateral"])
        assert 0.45 <= avg_fwd <= 0.55, f"forward avg = {avg_fwd:.2f}"
        assert 0.20 <= avg_bwd <= 0.30, f"backward avg = {avg_bwd:.2f}"
        assert 0.20 <= avg_lat <= 0.30, f"lateral avg = {avg_lat:.2f}"

    def test_within_cluster_edges_are_citation_kinds(self):
        run = fake_pipeline_run("topic", seed=42)
        cluster_of = {n.id: n.cluster for n in run.dag_node_objs}
        for e in run.dag_edge_objs:
            same_cluster = cluster_of[e.source] == cluster_of[e.target]
            if e.kind == "lateral":
                assert not same_cluster, \
                    f"lateral edge inside cluster: {e.source}→{e.target}"
            else:
                # forward/backward must be within-cluster
                assert same_cluster, \
                    f"{e.kind} edge crosses clusters: {e.source}→{e.target}"

    def test_dag_is_deterministic(self):
        a = fake_pipeline_run("topic", seed=42, n_ideas=12)
        b = fake_pipeline_run("topic", seed=42, n_ideas=12)
        assert [n.id for n in a.dag_node_objs] == [n.id for n in b.dag_node_objs]
        assert [(e.source, e.target, e.kind) for e in a.dag_edge_objs] == \
               [(e.source, e.target, e.kind) for e in b.dag_edge_objs]

    def test_dag_figure_has_expected_traces(self):
        run = fake_pipeline_run("topic", seed=42, n_ideas=12)
        fig = dag_figure(run)
        if fig is None:
            return
        # 1 node trace + up to 3 edge traces (one per kind that has any edges)
        assert 2 <= len(fig.data) <= 4
        # Last trace must be the node markers
        node_traces = [t for t in fig.data if t.mode == "markers"]
        assert len(node_traces) == 1
        assert len(node_traces[0].x) == len(run.dag_node_objs)

    def test_dag_figure_returns_none_for_empty_run(self):
        empty = FakeRun(topic="x", seed=0, dag_nodes=0, dag_edges=0)
        assert dag_figure(empty) is None


class TestAdminWiring:
    def test_simulator_function_exists(self):
        from pipeline_simulator import render_pipeline_simulator
        assert callable(render_pipeline_simulator)

    def test_admin_dashboard_imports_simulator(self):
        # Ensure the admin module wires the simulator correctly
        import admin_dashboard
        src = open(admin_dashboard.__file__, encoding="utf-8").read()
        assert "from pipeline_simulator import render_pipeline_simulator" in src
        assert "Pipeline Simulator" in src
