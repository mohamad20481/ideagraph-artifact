"""Tests for idea_simulator.py — visual simulation suite."""
import pytest
from idea_simulator import (
    extract_stages, simulate_outcomes, estimate_resources, run_simulation,
    build_method_flow, build_outcome_distribution, build_timeline,
    build_resource_gauges, _STAGE_KEYWORDS,
)


_SAMPLE_IDEA = {
    "title": "GNN for Drug Discovery",
    "method": "Use graph neural networks with attention to encode molecular structures. Train on ZINC benchmark dataset and evaluate on standard ADMET metrics with proper ablations.",
    "hypothesis": "Attention-based GNNs outperform fingerprint methods",
    "expected_outcome": "15% improvement in prediction accuracy",
    "resources": "8x A100 GPUs, 336 GPU-hours, 200GB data, 12 weeks",
    "quality_score": 0.72,
    "probe_scores": {"code": 0.8, "dataset": 0.7, "novelty": 0.6, "constraint": 0.6},
}


class TestStageExtraction:
    def test_returns_list_of_stages(self):
        stages = extract_stages(_SAMPLE_IDEA)
        assert isinstance(stages, list)
        assert len(stages) >= 3

    def test_starts_with_data(self):
        stages = extract_stages(_SAMPLE_IDEA)
        assert stages[0] == "📦 Data"

    def test_ends_with_evaluate(self):
        stages = extract_stages(_SAMPLE_IDEA)
        assert stages[-1] == "📊 Evaluate"

    def test_capped_at_seven(self):
        # Idea mentioning every keyword
        idea = {"method": " ".join(kw for kws, _ in _STAGE_KEYWORDS for kw in kws)}
        stages = extract_stages(idea)
        assert len(stages) <= 7

    def test_empty_idea_gives_default(self):
        stages = extract_stages({})
        assert "📦 Data" in stages
        assert "📊 Evaluate" in stages

    def test_no_duplicates(self):
        stages = extract_stages(_SAMPLE_IDEA)
        assert len(stages) == len(set(stages))


class TestOutcomeSimulation:
    def test_returns_required_fields(self):
        sim = simulate_outcomes(_SAMPLE_IDEA, n_trials=100)
        for key in ("trials", "mean", "std", "p10", "p50", "p90",
                     "success_pct", "baseline", "n_trials"):
            assert key in sim

    def test_n_trials_respected(self):
        sim = simulate_outcomes(_SAMPLE_IDEA, n_trials=50)
        assert len(sim["trials"]) == 50

    def test_percentile_ordering(self):
        sim = simulate_outcomes(_SAMPLE_IDEA, n_trials=200)
        assert sim["p10"] <= sim["p50"] <= sim["p90"]

    def test_outcomes_in_valid_range(self):
        sim = simulate_outcomes(_SAMPLE_IDEA, n_trials=100)
        for t in sim["trials"]:
            assert 0 <= t <= 100

    def test_deterministic_for_same_idea(self):
        a = simulate_outcomes(_SAMPLE_IDEA, n_trials=50)
        b = simulate_outcomes(_SAMPLE_IDEA, n_trials=50)
        assert a["trials"] == b["trials"]

    def test_different_ideas_different_outcomes(self):
        idea_a = {"title": "A", "quality_score": 0.3}
        idea_b = {"title": "B", "quality_score": 0.9}
        a = simulate_outcomes(idea_a, n_trials=50)
        b = simulate_outcomes(idea_b, n_trials=50)
        # Higher-quality should have higher mean
        assert b["mean"] > a["mean"]


class TestResourceEstimation:
    def test_extracts_gpu_hours_from_text(self):
        idea = {"resources": "8x A100, 336 GPU-hours"}
        res = estimate_resources(idea)
        assert res["gpu_hours"] == 336

    def test_extracts_data_size(self):
        idea = {"resources": "200GB dataset"}
        res = estimate_resources(idea)
        assert res["data_gb"] == 200.0

    def test_extracts_time(self):
        idea = {"resources": "12 weeks of training"}
        res = estimate_resources(idea)
        assert res["time_weeks"] == 12

    def test_falls_back_to_heuristic(self):
        idea = {"resources": "needs GPU compute"}  # no explicit number
        res = estimate_resources(idea)
        assert res["gpu_hours"] > 0  # heuristic kicks in

    def test_cost_computed(self):
        idea = {"resources": "100 GPU-hours"}
        res = estimate_resources(idea)
        assert res["cost_usd"] == 150.0  # $1.50/hour


class TestPlotlyFigures:
    def test_method_flow_returns_figure_or_none(self):
        fig = build_method_flow(_SAMPLE_IDEA)
        # Either a Plotly figure object or None (if plotly missing)
        assert fig is None or hasattr(fig, "to_dict")

    def test_outcome_distribution_returns_figure(self):
        fig = build_outcome_distribution(_SAMPLE_IDEA)
        assert fig is None or hasattr(fig, "to_dict")

    def test_timeline_returns_figure(self):
        fig = build_timeline(_SAMPLE_IDEA)
        assert fig is None or hasattr(fig, "to_dict")

    def test_timeline_uses_text_weeks(self):
        idea_short = {"resources": "2 weeks"}
        fig = build_timeline(idea_short)
        assert fig is None or hasattr(fig, "to_dict")

    def test_resource_gauges_returns_figure(self):
        fig = build_resource_gauges(_SAMPLE_IDEA)
        assert fig is None or hasattr(fig, "to_dict")


class TestUnifiedSimulation:
    def test_run_simulation_returns_all_components(self):
        sim = run_simulation(_SAMPLE_IDEA)
        for key in ("method_flow", "outcome_dist", "timeline", "resources",
                     "stages", "outcome_stats", "resource_stats"):
            assert key in sim

    def test_handles_minimal_idea(self):
        # Only title + method — should still produce a simulation
        sim = run_simulation({"title": "x", "method": "y"})
        assert sim["stages"]
        assert sim["outcome_stats"]["n_trials"] > 0

    def test_handles_empty_idea(self):
        sim = run_simulation({})
        assert "stages" in sim
        assert len(sim["stages"]) >= 2  # at least Data + Evaluate


class TestWhatIf:
    def test_more_compute_increases_mean(self):
        from idea_simulator import simulate_with_adjustments
        low = simulate_with_adjustments(_SAMPLE_IDEA, compute_multiplier=0.25, n_trials=100)
        high = simulate_with_adjustments(_SAMPLE_IDEA, compute_multiplier=4.0, n_trials=100)
        assert high["mean"] > low["mean"]

    def test_data_boost_increases_mean(self):
        from idea_simulator import simulate_with_adjustments
        worse = simulate_with_adjustments(_SAMPLE_IDEA, data_quality_boost=-0.2, n_trials=100)
        better = simulate_with_adjustments(_SAMPLE_IDEA, data_quality_boost=+0.2, n_trials=100)
        assert better["mean"] > worse["mean"]

    def test_returns_adjustments_metadata(self):
        from idea_simulator import simulate_with_adjustments
        sim = simulate_with_adjustments(_SAMPLE_IDEA, compute_multiplier=2.0, n_trials=50)
        assert "adjustments" in sim
        assert sim["adjustments"]["compute_multiplier"] == 2.0


class TestMultiOverlay:
    def test_returns_figure_and_summaries(self):
        from idea_simulator import build_multi_outcome_overlay
        result = build_multi_outcome_overlay([_SAMPLE_IDEA, {**_SAMPLE_IDEA, "title": "B"}])
        assert result is not None
        fig, summaries = result
        assert len(summaries) == 2
        for s in summaries:
            assert "p10" in s and "p50" in s and "p90" in s

    def test_caps_at_five(self):
        from idea_simulator import build_multi_outcome_overlay
        ideas = [{**_SAMPLE_IDEA, "title": f"Idea {i}"} for i in range(10)]
        _, summaries = build_multi_outcome_overlay(ideas)
        assert len(summaries) == 5

    def test_empty_returns_none(self):
        from idea_simulator import build_multi_outcome_overlay
        assert build_multi_outcome_overlay([]) is None


class TestSensitivity:
    def test_returns_sorted_factors(self):
        from idea_simulator import compute_sensitivity
        factors = compute_sensitivity(_SAMPLE_IDEA)
        assert len(factors) == 3
        # Sorted descending by range
        ranges = [f["range"] for f in factors]
        assert ranges == sorted(ranges, reverse=True)

    def test_each_factor_has_required_fields(self):
        from idea_simulator import compute_sensitivity
        for f in compute_sensitivity(_SAMPLE_IDEA):
            for key in ("factor", "low_p50", "high_p50", "base", "range",
                         "low_label", "high_label"):
                assert key in f


class TestParetoFrontier:
    def test_dominated_idea_excluded(self):
        from idea_simulator import compute_pareto_frontier
        ideas = [
            {"title": "Cheap+Good", "quality_score": 0.9, "resources": "100 GPU-hours"},
            {"title": "Expensive+Bad", "quality_score": 0.3, "resources": "500 GPU-hours"},
        ]
        opt = compute_pareto_frontier(ideas)
        assert opt[0] is True
        assert opt[1] is False

    def test_all_optimal_when_unique_tradeoffs(self):
        from idea_simulator import compute_pareto_frontier
        ideas = [
            {"title": "Cheap+OK", "quality_score": 0.5, "resources": "50 GPU-hours"},
            {"title": "Expensive+Great", "quality_score": 0.9, "resources": "500 GPU-hours"},
        ]
        opt = compute_pareto_frontier(ideas)
        assert all(opt)

    def test_empty_input(self):
        from idea_simulator import compute_pareto_frontier
        assert compute_pareto_frontier([]) == []


class TestRiskWaterfall:
    def test_returns_figure(self):
        from idea_simulator import build_risk_waterfall
        idea_with_fmea = {
            **_SAMPLE_IDEA,
            "_fmea": {
                "failure_modes": [
                    {"mode": "Data leakage", "severity": 5, "detectability": 4,
                     "risk_priority": 20, "mitigation": "stratify"},
                    {"mode": "Overfitting", "severity": 3, "detectability": 3,
                     "risk_priority": 9, "mitigation": "regularize"},
                ],
            },
        }
        fig = build_risk_waterfall(idea_with_fmea)
        assert fig is None or hasattr(fig, "to_dict")

    def test_handles_idea_without_fmea(self):
        from idea_simulator import build_risk_waterfall
        # No _fmea field — should fall back to heuristic
        fig = build_risk_waterfall(_SAMPLE_IDEA)
        assert fig is None or hasattr(fig, "to_dict")


class TestExecutionPlayback:
    def test_returns_animated_figure(self):
        from idea_simulator import build_execution_playback
        fig = build_execution_playback(_SAMPLE_IDEA)
        assert fig is None or hasattr(fig, "to_dict")
        if fig:
            # Should have at least one frame for animation
            assert len(fig.frames) >= 2

    def test_handles_minimal_idea(self):
        from idea_simulator import build_execution_playback
        fig = build_execution_playback({"method": "x"})
        assert fig is None or hasattr(fig, "to_dict")


class TestConfidenceCone:
    def test_returns_figure(self):
        from idea_simulator import build_confidence_cone
        fig = build_confidence_cone(_SAMPLE_IDEA)
        assert fig is None or hasattr(fig, "to_dict")

    def test_handles_short_timeline(self):
        from idea_simulator import build_confidence_cone
        idea = {**_SAMPLE_IDEA, "resources": "2 weeks"}
        fig = build_confidence_cone(idea)
        assert fig is None or hasattr(fig, "to_dict")


class TestBudgetBurndown:
    def test_returns_figure(self):
        from idea_simulator import build_budget_burndown
        fig = build_budget_burndown(_SAMPLE_IDEA)
        assert fig is None or hasattr(fig, "to_dict")

    def test_uses_resource_data(self):
        from idea_simulator import build_budget_burndown
        # Idea with explicit resources
        idea = {"resources": "100 GPU-hours, 4 weeks"}
        fig = build_budget_burndown(idea)
        assert fig is None or hasattr(fig, "to_dict")


class TestThreeDIdeaSpace:
    def test_returns_figure_for_multiple_ideas(self):
        from idea_simulator import build_3d_idea_space
        ideas = [
            _SAMPLE_IDEA,
            {**_SAMPLE_IDEA, "title": "B", "quality_score": 0.4},
            {**_SAMPLE_IDEA, "title": "C", "quality_score": 0.9},
        ]
        fig = build_3d_idea_space(ideas)
        assert fig is None or hasattr(fig, "to_dict")

    def test_highlights_chosen_idea(self):
        from idea_simulator import build_3d_idea_space
        ideas = [_SAMPLE_IDEA, {**_SAMPLE_IDEA, "title": "B"}]
        fig = build_3d_idea_space(ideas, highlight_idx=1)
        assert fig is None or hasattr(fig, "to_dict")

    def test_empty_returns_none(self):
        from idea_simulator import build_3d_idea_space
        assert build_3d_idea_space([]) is None


class TestProbeSunburst:
    def test_returns_figure_with_scores(self):
        from idea_simulator import build_probe_sunburst
        fig = build_probe_sunburst(_SAMPLE_IDEA)
        assert fig is None or hasattr(fig, "to_dict")

    def test_returns_none_when_no_probe_scores(self):
        from idea_simulator import build_probe_sunburst
        idea = {"title": "no probes"}
        fig = build_probe_sunburst(idea)
        assert fig is None

    def test_handles_partial_scores(self):
        from idea_simulator import build_probe_sunburst
        idea = {"probe_scores": {"code": 0.7, "novelty": 0.5}}  # only 2 of 10
        fig = build_probe_sunburst(idea)
        assert fig is None or hasattr(fig, "to_dict")


class TestCarbonFootprint:
    def test_more_compute_more_co2(self):
        from idea_simulator import estimate_carbon
        small = estimate_carbon({"resources": "20 GPU-hours"})
        big = estimate_carbon({"resources": "1000 GPU-hours"})
        assert big["kg_co2"] > small["kg_co2"]

    def test_green_score_inverse_of_co2(self):
        from idea_simulator import estimate_carbon
        small = estimate_carbon({"resources": "5 GPU-hours"})
        big = estimate_carbon({"resources": "5000 GPU-hours"})
        assert small["green_score"] > big["green_score"]

    def test_returns_required_fields(self):
        from idea_simulator import estimate_carbon
        c = estimate_carbon(_SAMPLE_IDEA)
        for key in ("gpu_hours", "kwh", "kg_co2", "miles_driven_eq",
                    "trees_year_eq", "household_days_eq", "green_score"):
            assert key in c

    def test_green_score_in_range(self):
        from idea_simulator import estimate_carbon
        c = estimate_carbon(_SAMPLE_IDEA)
        assert 0 <= c["green_score"] <= 100

    def test_figure_renders(self):
        from idea_simulator import build_carbon_footprint
        fig = build_carbon_footprint(_SAMPLE_IDEA)
        assert fig is None or hasattr(fig, "to_dict")


class TestCitationForecast:
    def test_higher_quality_more_citations(self):
        from idea_simulator import forecast_citations
        low_q = forecast_citations({"quality_score": 0.3, "probe_scores": {}})
        high_q = forecast_citations({"quality_score": 0.9, "probe_scores": {}})
        assert high_q["asymptote"] > low_q["asymptote"]

    def test_cumulative_monotonic(self):
        from idea_simulator import forecast_citations
        f = forecast_citations(_SAMPLE_IDEA)
        # Cumulative citations must never decrease year over year
        for i in range(1, 5):
            assert f["cumulative"][i] >= f["cumulative"][i - 1]

    def test_quality_tier_categorical(self):
        from idea_simulator import forecast_citations
        f = forecast_citations(_SAMPLE_IDEA)
        assert f["quality_tier"] in ("viral", "strong", "moderate", "minor")

    def test_h_index_at_least_1(self):
        from idea_simulator import forecast_citations
        f = forecast_citations(_SAMPLE_IDEA)
        assert f["h_index_contrib"] >= 1

    def test_chart_renders(self):
        from idea_simulator import build_citation_forecast
        fig = build_citation_forecast(_SAMPLE_IDEA)
        assert fig is None or hasattr(fig, "to_dict")


class TestSimilarityNetwork:
    def test_similar_methods_score_high(self):
        from idea_simulator import _idea_similarity
        a = {"title": "GNN", "method": "graph neural networks message passing molecular"}
        b = {"title": "GNN2", "method": "graph neural networks attention molecular"}
        assert _idea_similarity(a, b) > 0.3

    def test_unrelated_methods_score_low(self):
        from idea_simulator import _idea_similarity
        a = {"title": "ML", "method": "graph neural networks molecular"}
        b = {"title": "RL", "method": "reinforcement learning robotic arm"}
        assert _idea_similarity(a, b) < 0.2

    def test_network_handles_few_ideas(self):
        from idea_simulator import build_similarity_network
        # Single idea — can't build a network
        assert build_similarity_network([{"title": "x", "method": "y"}]) is None

    def test_network_renders_for_multiple(self):
        from idea_simulator import build_similarity_network
        ideas = [_SAMPLE_IDEA, {**_SAMPLE_IDEA, "title": "B"}]
        fig = build_similarity_network(ideas)
        assert fig is None or hasattr(fig, "to_dict")


class TestSuccessFunnel:
    def test_returns_figure(self):
        from idea_simulator import build_success_funnel
        fig = build_success_funnel(_SAMPLE_IDEA)
        assert fig is None or hasattr(fig, "to_dict")

    def test_handles_no_probe_scores(self):
        from idea_simulator import build_success_funnel
        fig = build_success_funnel({"title": "x", "quality_score": 0.5})
        assert fig is None or hasattr(fig, "to_dict")


class TestStageCriticality:
    def test_returns_figure(self):
        from idea_simulator import build_stage_criticality
        fig = build_stage_criticality(_SAMPLE_IDEA)
        assert fig is None or hasattr(fig, "to_dict")

    def test_handles_empty_probes(self):
        from idea_simulator import build_stage_criticality
        fig = build_stage_criticality({"probe_scores": {}})
        assert fig is None or hasattr(fig, "to_dict")


class TestNoveltyConstellation:
    def test_returns_figure(self):
        from idea_simulator import build_novelty_constellation
        fig = build_novelty_constellation(_SAMPLE_IDEA, [], [])
        assert fig is None or hasattr(fig, "to_dict")

    def test_with_other_ideas_and_papers(self):
        from idea_simulator import build_novelty_constellation
        fig = build_novelty_constellation(
            _SAMPLE_IDEA,
            [{**_SAMPLE_IDEA, "title": "Other"}],
            [{"title": "BERT paper"}, {"title": "CNN paper"}],
        )
        assert fig is None or hasattr(fig, "to_dict")

    def test_deterministic_position(self):
        from idea_simulator import _hash_to_2d
        a = _hash_to_2d("test idea")
        b = _hash_to_2d("test idea")
        assert a == b
        c = _hash_to_2d("different idea")
        assert c != a


class TestDNAFingerprint:
    def test_returns_figure(self):
        from idea_simulator import build_dna_fingerprint
        fig = build_dna_fingerprint(_SAMPLE_IDEA)
        assert fig is None or hasattr(fig, "to_dict")

    def test_bands_count_is_16(self):
        from idea_simulator import _idea_dna_bands
        bands = _idea_dna_bands(_SAMPLE_IDEA)
        assert len(bands) == 16
        for color, intensity in bands:
            assert color.startswith("#")
            assert 0 <= intensity <= 1

    def test_same_idea_same_fingerprint(self):
        from idea_simulator import _idea_dna_bands
        a = _idea_dna_bands(_SAMPLE_IDEA)
        b = _idea_dna_bands(_SAMPLE_IDEA)
        assert a == b


class TestTimeMachine:
    def test_detects_known_techniques(self):
        from idea_simulator import detect_techniques
        idea = {"method": "Use transformer with attention and BERT-style pretraining"}
        techs = detect_techniques(idea)
        terms = [t["term"] for t in techs]
        assert "transformer" in terms
        assert "attention" in terms

    def test_techniques_sorted_by_year(self):
        from idea_simulator import detect_techniques
        techs = detect_techniques({
            "method": "Combine LSTM with transformer and attention mechanism",
        })
        years = [t["year"] for t in techs]
        assert years == sorted(years)

    def test_unknown_idea_returns_empty(self):
        from idea_simulator import detect_techniques
        assert detect_techniques({"method": "purely fictional gobbledygook xyzqwerty"}) == []

    def test_no_techniques_returns_none_chart(self):
        from idea_simulator import build_time_machine
        fig = build_time_machine({"method": "no known terms here"})
        assert fig is None

    def test_chart_renders_with_techniques(self):
        from idea_simulator import build_time_machine
        idea = {"method": "transformer attention BERT"}
        fig = build_time_machine(idea)
        assert fig is None or hasattr(fig, "to_dict")


class TestReviewerChat:
    def test_returns_four_messages(self):
        from idea_simulator import simulate_reviewer_chat
        chat = simulate_reviewer_chat(_SAMPLE_IDEA)
        # 1 meta-reviewer + 3 reviewers
        assert len(chat) == 4

    def test_has_meta_reviewer_first(self):
        from idea_simulator import simulate_reviewer_chat
        chat = simulate_reviewer_chat(_SAMPLE_IDEA)
        assert "Meta" in chat[0]["role"]

    def test_negative_for_low_quality(self):
        from idea_simulator import simulate_reviewer_chat
        bad = {**_SAMPLE_IDEA, "probe_scores": {k: 0.1 for k in
                                                  _SAMPLE_IDEA["probe_scores"]}}
        chat = simulate_reviewer_chat(bad)
        assert chat[0]["msg"].startswith("❌") or "Reject" in chat[0]["msg"]

    def test_positive_for_high_quality(self):
        from idea_simulator import simulate_reviewer_chat
        # All 10 probe dimensions must be high to trigger positive verdict
        all_probes = ["code", "dataset", "constraint", "scalability",
                      "specificity", "clarity", "testability", "risk_balance",
                      "novelty", "significance"]
        good = {**_SAMPLE_IDEA, "probe_scores": {k: 0.9 for k in all_probes}}
        chat = simulate_reviewer_chat(good)
        assert "Accept" in chat[0]["msg"] or chat[0]["msg"].startswith("✅")


class TestTarot:
    def test_returns_three_cards(self):
        from idea_simulator import generate_tarot
        cards = generate_tarot(_SAMPLE_IDEA)
        assert len(cards) == 3

    def test_each_card_has_required_fields(self):
        from idea_simulator import generate_tarot
        for card in generate_tarot(_SAMPLE_IDEA):
            assert "name" in card
            assert "meaning" in card
            assert "narrative" in card

    def test_html_contains_all_cards(self):
        from idea_simulator import generate_tarot, tarot_to_html
        cards = generate_tarot(_SAMPLE_IDEA)
        html = tarot_to_html(cards)
        for card in cards:
            assert card["name"] in html

    def test_low_quality_gets_warning_card(self):
        from idea_simulator import generate_tarot
        bad = {"quality_score": 0.2, "probe_scores": {}}
        cards = generate_tarot(bad)
        # Future should warn for low quality
        assert "Mist" in cards[1]["name"] or "Crossroads" in cards[2]["name"]


class TestPokemonCard:
    def test_returns_html(self):
        from idea_simulator import build_pokemon_card
        html = build_pokemon_card(_SAMPLE_IDEA)
        assert "<div" in html
        assert "HP" in html and "ATK" in html and "DEF" in html

    def test_contains_idea_title(self):
        from idea_simulator import build_pokemon_card
        html = build_pokemon_card(_SAMPLE_IDEA)
        assert "GNN for Drug Discovery" in html

    def test_includes_rarity_badge(self):
        from idea_simulator import build_pokemon_card
        # High-quality should be rare/epic
        good = {**_SAMPLE_IDEA, "quality_score": 0.95,
                "probe_scores": {k: 0.95 for k in
                                  ["code", "dataset", "constraint", "scalability",
                                   "specificity", "clarity", "testability",
                                   "risk_balance", "novelty", "significance"]}}
        html = build_pokemon_card(good)
        assert "Legendary" in html or "Epic" in html

    def test_handles_missing_methodology(self):
        from idea_simulator import build_pokemon_card
        idea = {"title": "x", "quality_score": 0.5}
        html = build_pokemon_card(idea)
        assert "<div" in html  # falls back gracefully


class TestWeatherForecast:
    def test_returns_seven_weeks(self):
        from idea_simulator import build_weather_forecast
        forecast = build_weather_forecast(_SAMPLE_IDEA)
        assert len(forecast) == 7

    def test_each_week_has_required_fields(self):
        from idea_simulator import build_weather_forecast
        for w in build_weather_forecast(_SAMPLE_IDEA):
            for key in ("week", "phase", "risk", "icon", "label", "color", "advice"):
                assert key in w

    def test_html_renders_all_weeks(self):
        from idea_simulator import build_weather_forecast, weather_to_html
        forecast = build_weather_forecast(_SAMPLE_IDEA)
        html = weather_to_html(forecast)
        # All 7 weeks should show up
        for w in forecast:
            assert f"Week {w['week']}" in html

    def test_high_quality_more_sunny(self):
        from idea_simulator import build_weather_forecast
        good = {**_SAMPLE_IDEA, "quality_score": 0.95,
                "probe_scores": {k: 0.95 for k in
                                  ["code", "dataset", "specificity", "risk_balance"]}}
        bad = {**_SAMPLE_IDEA, "quality_score": 0.2,
               "probe_scores": {k: 0.1 for k in
                                 ["code", "dataset", "specificity", "risk_balance"]}}
        good_sunny = sum(1 for f in build_weather_forecast(good) if f["risk"] < 0.4)
        bad_sunny = sum(1 for f in build_weather_forecast(bad) if f["risk"] < 0.4)
        assert good_sunny >= bad_sunny


class TestTwinUniverse:
    def test_figure_renders(self):
        from idea_simulator import build_twin_universe
        fig = build_twin_universe(_SAMPLE_IDEA)
        assert fig is None or hasattr(fig, "to_dict")

    def test_summary_returns_four_universes(self):
        from idea_simulator import twin_universe_summary
        summary = twin_universe_summary(_SAMPLE_IDEA)
        assert len(summary) == 4

    def test_each_universe_has_verdict(self):
        from idea_simulator import twin_universe_summary
        for u in twin_universe_summary(_SAMPLE_IDEA):
            assert u["verdict"] in ("🚀 Better", "⚖️ Similar", "📉 Worse")
            assert "p50" in u and "delta" in u


class TestOriginStory:
    def test_returns_html_string(self):
        from idea_simulator import generate_origin_story
        story = generate_origin_story(_SAMPLE_IDEA)
        assert isinstance(story, str)
        assert "<p" in story  # has paragraphs

    def test_three_paragraphs(self):
        from idea_simulator import generate_origin_story
        story = generate_origin_story(_SAMPLE_IDEA)
        # Should have 3 <p> blocks
        assert story.count("<p") == 3

    def test_high_novelty_pioneer_language(self):
        from idea_simulator import generate_origin_story
        idea = {**_SAMPLE_IDEA, "probe_scores": {"novelty": 0.9, "significance": 0.5}}
        story = generate_origin_story(idea)
        assert "departure" in story or "novelty" in story.lower() or "novel" in story.lower()

    def test_includes_idea_title(self):
        from idea_simulator import generate_origin_story
        story = generate_origin_story(_SAMPLE_IDEA)
        assert "GNN for Drug Discovery" in story


class TestProbabilityCloud:
    def test_returns_figure(self):
        from idea_simulator import build_probability_cloud
        fig = build_probability_cloud(_SAMPLE_IDEA)
        assert fig is None or hasattr(fig, "to_dict")


class TestMovieTrailer:
    def test_returns_required_fields(self):
        from idea_simulator import generate_movie_trailer
        t = generate_movie_trailer(_SAMPLE_IDEA)
        for k in ("title", "genre", "tagline", "pitch", "cast", "rating", "stars", "release"):
            assert k in t

    def test_high_quality_gets_high_stars(self):
        from idea_simulator import generate_movie_trailer
        idea = {**_SAMPLE_IDEA, "quality_score": 0.85,
                "probe_scores": {"novelty": 0.85, "significance": 0.85}}
        assert generate_movie_trailer(idea)["stars"] == 5

    def test_low_quality_gets_low_stars(self):
        from idea_simulator import generate_movie_trailer
        idea = {**_SAMPLE_IDEA, "quality_score": 0.2}
        assert generate_movie_trailer(idea)["stars"] == 2

    def test_deterministic(self):
        from idea_simulator import generate_movie_trailer
        a = generate_movie_trailer(_SAMPLE_IDEA)
        b = generate_movie_trailer(_SAMPLE_IDEA)
        assert a["tagline"] == b["tagline"] and a["cast"] == b["cast"]

    def test_html_contains_title(self):
        from idea_simulator import generate_movie_trailer, trailer_to_html
        t = generate_movie_trailer(_SAMPLE_IDEA)
        html = trailer_to_html(t)
        assert _SAMPLE_IDEA["title"] in html and "<div" in html


class TestQuestLog:
    def test_includes_main_quest(self):
        from idea_simulator import generate_quest_log
        quests = generate_quest_log(_SAMPLE_IDEA)
        assert any("Main" in q["type"] for q in quests)

    def test_weak_probes_become_side_quests(self):
        from idea_simulator import generate_quest_log
        idea = {**_SAMPLE_IDEA, "quality_score": 0.5,
                "probe_scores": {"code": 0.2, "dataset": 0.2, "novelty": 0.2,
                                 "specificity": 0.9, "clarity": 0.9}}
        quests = generate_quest_log(idea)
        side = [q for q in quests if "Side" in q["type"]]
        assert len(side) >= 1

    def test_high_quality_unlocks_endgame(self):
        from idea_simulator import generate_quest_log
        idea = {**_SAMPLE_IDEA, "quality_score": 0.85}
        quests = generate_quest_log(idea)
        assert any("Endgame" in q["type"] for q in quests)

    def test_low_quality_locks_main_quest(self):
        from idea_simulator import generate_quest_log
        idea = {**_SAMPLE_IDEA, "quality_score": 0.2, "probe_scores": {}}
        quests = generate_quest_log(idea)
        main = next(q for q in quests if "Main" in q["type"])
        assert main["status"] == "locked"

    def test_html_renders(self):
        from idea_simulator import generate_quest_log, quest_log_to_html
        html = quest_log_to_html(generate_quest_log(_SAMPLE_IDEA))
        assert "<div" in html and ("ACTIVE" in html or "LOCKED" in html)


class TestConferenceMatch:
    def test_returns_sorted_descending(self):
        from idea_simulator import match_conferences
        m = match_conferences(_SAMPLE_IDEA)
        assert len(m) >= 3
        for i in range(len(m) - 1):
            assert m[i]["match_score"] >= m[i + 1]["match_score"]

    def test_required_fields(self):
        from idea_simulator import match_conferences
        m = match_conferences(_SAMPLE_IDEA)
        for entry in m:
            for k in ("name", "tier", "deadline", "acceptance_rate",
                      "match_score", "verdict"):
                assert k in entry

    def test_high_quality_gets_strong_fit(self):
        from idea_simulator import match_conferences
        idea = {**_SAMPLE_IDEA, "quality_score": 0.95,
                "probe_scores": {k: 0.95 for k in
                                 ("novelty", "significance", "clarity",
                                  "testability", "specificity", "scalability",
                                  "code", "dataset", "constraint", "risk_balance")}}
        m = match_conferences(idea)
        assert any("Strong" in entry["verdict"] for entry in m)

    def test_low_quality_below_bar(self):
        from idea_simulator import match_conferences
        idea = {**_SAMPLE_IDEA, "quality_score": 0.1,
                "probe_scores": {k: 0.1 for k in
                                 ("novelty", "significance", "clarity",
                                  "testability", "specificity", "scalability")}}
        m = match_conferences(idea)
        assert any("Below" in entry["verdict"] for entry in m)

    def test_html_renders(self):
        from idea_simulator import match_conferences, conference_match_to_html
        html = conference_match_to_html(match_conferences(_SAMPLE_IDEA))
        assert "🥇" in html and "Match" in html


class TestIdeaMosaic:
    def test_returns_figure(self):
        from idea_simulator import build_idea_mosaic
        fig = build_idea_mosaic(_SAMPLE_IDEA)
        assert fig is None or hasattr(fig, "to_dict")

    def test_deterministic(self):
        from idea_simulator import build_idea_mosaic
        a = build_idea_mosaic(_SAMPLE_IDEA)
        b = build_idea_mosaic(_SAMPLE_IDEA)
        if a is None or b is None:
            return
        assert a.to_dict() == b.to_dict()

    def test_different_ideas_differ(self):
        from idea_simulator import build_idea_mosaic
        a = build_idea_mosaic(_SAMPLE_IDEA)
        b = build_idea_mosaic({**_SAMPLE_IDEA, "title": "Totally Different Title"})
        if a is None or b is None:
            return
        assert a.to_dict() != b.to_dict()


class TestAcceptanceSpeech:
    def test_returns_three_paragraphs(self):
        from idea_simulator import generate_acceptance_speech
        s = generate_acceptance_speech(_SAMPLE_IDEA)
        assert s.count("<p") == 3

    def test_contains_title(self):
        from idea_simulator import generate_acceptance_speech
        s = generate_acceptance_speech(_SAMPLE_IDEA)
        assert _SAMPLE_IDEA["title"] in s

    def test_high_novelty_uses_long_shot_language(self):
        from idea_simulator import generate_acceptance_speech
        idea = {**_SAMPLE_IDEA, "probe_scores": {"novelty": 0.9}}
        s = generate_acceptance_speech(idea)
        assert "long shot" in s.lower() or "didn't prepare" in s.lower()

    def test_deterministic(self):
        from idea_simulator import generate_acceptance_speech
        a = generate_acceptance_speech(_SAMPLE_IDEA)
        b = generate_acceptance_speech(_SAMPLE_IDEA)
        assert a == b
