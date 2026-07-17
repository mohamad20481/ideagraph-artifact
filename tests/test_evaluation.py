"""
Deterministic tests for the evaluation harness (no network, no torch, no LLM).

Every numeric assertion below is an anchor computed by hand — if one fails,
the metric/statistic definition has drifted from evaluation/README.md.
"""
import json
import math

import numpy as np
import pytest

from evaluation import EvaluationError
from evaluation.metrics import (
    idea_to_text, diversity_index, div_pair, ras, ras_topic, nft, nft_idea,
)
from evaluation.tost import (
    noninferiority_test, tost_equivalence, discriminability, clopper_pearson,
)
from evaluation.crossval import (
    CrossValEntry, from_dict, save_jsonl, load_jsonl, blind, unblind_scores,
    TUPLE_FIELDS,
)
from evaluation.judges import Judge, rate_all, PanelRatings, DIMENSIONS


# ── Stub embedder: title marker word → fixed vector ─────────────────────────

_VECS = {
    "alpha": np.array([1.0, 0.0, 0.0]),
    "beta": np.array([0.0, 1.0, 0.0]),
    "gamma": np.array([0.0, 0.0, 1.0]),
    "diag": np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0),
    "anti": np.array([-1.0, 0.0, 0.0]),
    "mix": np.array([0.6, 0.8, 0.0]),
}


class StubEmbedder:
    def encode(self, texts, **kw):
        rows = []
        for t in texts:
            for marker, vec in _VECS.items():
                if marker in t:
                    rows.append(vec)
                    break
            else:
                raise AssertionError(f"stub embedder: no marker in {t!r}")
        return np.vstack(rows)


def _idea(marker, **extra):
    d = {"title": marker, "method": f"method of {marker}"}
    d.update(extra)
    return d


# ── metrics: text extraction ─────────────────────────────────────────────────

def test_idea_to_text_field_order_and_empty():
    txt = idea_to_text({"method": "M", "title": "T", "motivation": "W"})
    assert txt == "T. W. M"          # canonical order: title, motivation, method
    with pytest.raises(EvaluationError):
        idea_to_text({"title": "  ", "irrelevant": "x"})
    with pytest.raises(EvaluationError):
        idea_to_text("not a dict")


# ── metrics: DI / Div-Pair ───────────────────────────────────────────────────

def test_di_orthogonal_is_one_and_identical_is_zero():
    emb = StubEmbedder()
    assert diversity_index(
        [_idea("alpha"), _idea("beta"), _idea("gamma")], embedder=emb
    ) == pytest.approx(1.0)
    assert diversity_index(
        [_idea("alpha"), _idea("alpha"), _idea("alpha")], embedder=emb
    ) == pytest.approx(0.0)


def test_di_known_mixed_value():
    # pairs: (alpha,beta) d=1; (alpha,diag) d=1-√2/2; (beta,diag) d=1-√2/2
    emb = StubEmbedder()
    expected = (1.0 + 2 * (1.0 - math.sqrt(2) / 2)) / 3.0
    got = diversity_index([_idea("alpha"), _idea("beta"), _idea("diag")],
                          embedder=emb)
    assert got == pytest.approx(expected, abs=1e-9)
    # Div-Pair is the identical computation
    assert div_pair([_idea("alpha"), _idea("beta"), _idea("diag")],
                    embedder=emb) == pytest.approx(expected, abs=1e-9)


def test_di_requires_two_ideas():
    with pytest.raises(EvaluationError):
        diversity_index([_idea("alpha")], embedder=StubEmbedder())


# ── metrics: RAS ─────────────────────────────────────────────────────────────

def test_ras_max_vs_mean_and_floor():
    emb = StubEmbedder()
    ideas = [_idea("alpha"), _idea("mix")]      # cos vs alpha-ref: 1.0, 0.6
    assert ras_topic(ideas, "reference alpha", agg="max",
                     embedder=emb) == pytest.approx(1.0)
    assert ras_topic(ideas, "reference alpha", agg="mean",
                     embedder=emb) == pytest.approx(0.8)
    # negative cosine floors at 0
    assert ras_topic([_idea("anti")], "reference alpha",
                     embedder=emb) == pytest.approx(0.0)
    with pytest.raises(EvaluationError):
        ras_topic(ideas, "reference alpha", agg="median", embedder=emb)


def test_ras_benchmark_mean_and_missing_reference():
    emb = StubEmbedder()
    per_topic = {"t1": [_idea("alpha")], "t2": [_idea("mix")]}
    refs = {"t1": "ref alpha", "t2": "ref alpha"}
    # t1: cos=1.0 ; t2: cos=0.6 → mean 0.8
    assert ras(per_topic, refs, agg="max",
               embedder=emb) == pytest.approx(0.8)
    with pytest.raises(EvaluationError):
        ras(per_topic, {"t1": "ref alpha"}, embedder=emb)


# ── metrics: NFT ─────────────────────────────────────────────────────────────

def test_nft_harmonic_anchors():
    assert nft_idea(0.5, 0.5) == pytest.approx(0.5)
    assert nft_idea(1.0, 0.0) == pytest.approx(0.0)   # tradeoff punishes imbalance
    assert nft_idea(0.0, 0.0) == pytest.approx(0.0)
    assert nft_idea(0.8, 0.4) == pytest.approx(2 * 0.8 * 0.4 / 1.2, abs=1e-12)
    with pytest.raises(EvaluationError):
        nft_idea(1.2, 0.5)
    # set-level = mean of per-idea harmonics (NOT harmonic of set means)
    ideas = [{"nov": 1.0, "feas": 0.0}, {"nov": 0.5, "feas": 0.5}]
    assert nft(ideas) == pytest.approx(0.25)


# ── tost: numeric anchors ────────────────────────────────────────────────────

def test_ni_boundary_diff_equals_minus_delta_gives_p_half():
    test = [2.7, 3.3]     # mean 3.0
    ref = [3.0, 3.6]      # mean 3.3 → d = -0.3 = -delta → t = 0 → p = 0.5
    r = noninferiority_test(test, ref, delta=0.3)
    assert r.diff == pytest.approx(-0.3, abs=1e-12)
    assert r.t_stat == pytest.approx(0.0, abs=1e-12)
    assert r.p_value == pytest.approx(0.5, abs=1e-9)
    assert not r.established


def test_ni_clearly_noninferior_and_clearly_inferior():
    rng = np.random.RandomState(7)
    base = rng.normal(0, 0.05, size=100)
    same = (3.0 + base).tolist()
    r = noninferiority_test(same, (3.0 + rng.normal(0, 0.05, 100)).tolist(),
                            delta=0.3)
    assert r.p_value < 1e-6 and r.established
    worse = (2.0 + rng.normal(0, 0.05, 100)).tolist()
    r2 = noninferiority_test(worse, same, delta=0.3)
    assert r2.p_value > 0.999 and not r2.established


def test_tost_equivalence_directions():
    rng = np.random.RandomState(11)
    a = (3.0 + rng.normal(0, 0.05, 80)).tolist()
    b = (3.0 + rng.normal(0, 0.05, 80)).tolist()
    eq = tost_equivalence(a, b, delta=0.3)
    assert eq.equivalent and eq.p_tost < 1e-4
    c = (3.5 + rng.normal(0, 0.05, 80)).tolist()
    ne = tost_equivalence(c, b, delta=0.3)   # diff +0.5 > delta → not equivalent
    assert not ne.equivalent


def test_tost_degenerate_inputs_raise():
    with pytest.raises(EvaluationError):
        noninferiority_test([1.0], [1.0, 2.0], delta=0.3)      # n < 2
    with pytest.raises(EvaluationError):
        noninferiority_test([1.0, 1.0], [2.0, 2.0], delta=0.3)  # zero SE
    with pytest.raises(EvaluationError):
        noninferiority_test([1.0, 2.0], [1.0, 2.0], delta=-0.3)


def test_discriminability_anchors():
    chance = discriminability(150, 300)
    assert chance.accuracy == pytest.approx(0.5)
    assert chance.p_two_sided > 0.9
    above = discriminability(172, 300)          # 57.3%
    assert above.accuracy == pytest.approx(172 / 300)
    assert above.p_two_sided < 0.05
    assert above.ci_low < above.accuracy < above.ci_high
    lo, hi = clopper_pearson(0, 20)
    assert lo == 0.0 and 0.0 < hi < 0.25
    lo2, hi2 = clopper_pearson(20, 20)
    assert hi2 == 1.0 and 0.75 < lo2 < 1.0


# ── crossval: schema, IO, blinding ───────────────────────────────────────────

def _entry(i, source="generated", **over):
    fields = {f: f"{f} text {i}" for f in TUPLE_FIELDS}
    prov = {"run_id": "r1"} if source == "generated" else \
           {"paper_title": f"P{i}", "venue": "ACL", "year": 2025}
    fields.update(over)
    return CrossValEntry(id=f"e{i}", source=source, topic=f"topic {i % 3}",
                         provenance=prov, **fields)


def test_crossval_validation():
    _entry(1).validate()                                  # ok
    with pytest.raises(EvaluationError):                  # empty required field
        _entry(2, motivation="  ").validate()
    with pytest.raises(EvaluationError):                  # bad source
        CrossValEntry(id="x", source="synthetic", topic="t",
                      **{f: "v" for f in TUPLE_FIELDS}).validate()
    with pytest.raises(EvaluationError):                  # published w/o provenance
        CrossValEntry(id="x", source="published", topic="t",
                      **{f: "v" for f in TUPLE_FIELDS}).validate()


def test_crossval_jsonl_roundtrip(tmp_path):
    entries = [_entry(i) for i in range(3)] + \
              [_entry(i + 10, source="published") for i in range(3)]
    path = str(tmp_path / "cv.jsonl")
    save_jsonl(entries, path)
    loaded = load_jsonl(path)
    assert [e.id for e in loaded] == [e.id for e in entries]
    assert loaded[3].provenance["venue"] == "ACL"
    with pytest.raises(EvaluationError):                  # duplicate ids
        save_jsonl([_entry(1), _entry(1)], str(tmp_path / "dup.jsonl"))


def test_blinding_strips_identity_and_is_deterministic():
    entries = [_entry(i) for i in range(4)] + \
              [_entry(i + 10, source="published") for i in range(4)]
    view, key = blind(entries, seed=42)
    assert len(view) == 8 and len(key) == 8
    for item in view:
        assert "source" not in item and "provenance" not in item
        assert "id" not in item and "title" not in item
        assert all(item[f] for f in TUPLE_FIELDS)
    view2, key2 = blind(entries, seed=42)
    assert view == view2 and key == key2                  # seed-stable
    view3, _ = blind(entries, seed=43)
    assert view != view3                                  # seed matters
    # unblind maps back to the right sources
    scores = {b: {"overall": 3} for b in key}
    un = unblind_scores(scores, key, entries)
    assert sum(1 for v in un.values() if v["source"] == "published") == 4


# ── judges: aggregation with stub clients ────────────────────────────────────

class _Resp:
    def __init__(self, text, success=True):
        self.text = text
        self.success = success


class StubJudgeClient:
    def __init__(self, scores):
        self._scores = scores

    def call(self, **kw):
        return _Resp(json.dumps(self._scores))


def _blind_item(i):
    item = {"blind_id": f"b{i:03d}", "topic": "t"}
    item.update({f: f"{f} {i}" for f in TUPLE_FIELDS})
    return item


def test_panel_aggregation_mean_and_unit():
    j1 = Judge("j1", StubJudgeClient({d: 4 for d in DIMENSIONS}))
    j2 = Judge("j2", StubJudgeClient({d: 5 for d in DIMENSIONS}))
    ratings = rate_all([_blind_item(1), _blind_item(2)], [j1, j2])
    m = ratings.mean_scores("b001")
    assert m["novelty"] == pytest.approx(4.5)
    u = ratings.unit_scores("b001")
    assert u["novelty"] == pytest.approx((4.5 - 1) / 4)   # 0.875
    sample = ratings.dimension_sample(["b001", "b002"], "overall")
    assert sample == [pytest.approx(4.5)] * 2


def test_judge_out_of_range_and_failure_raise():
    bad = Judge("bad", StubJudgeClient({**{d: 3 for d in DIMENSIONS},
                                        "novelty": 7}))
    with pytest.raises(EvaluationError):
        rate_all([_blind_item(1)], [bad])

    class FailingClient:
        def call(self, **kw):
            return _Resp("", success=False)

    with pytest.raises(EvaluationError):
        rate_all([_blind_item(1)], [Judge("f", FailingClient())])

    with pytest.raises(EvaluationError):                  # duplicate names
        rate_all([_blind_item(1)], [Judge("x", StubJudgeClient({d: 3 for d in DIMENSIONS})),
                                    Judge("x", StubJudgeClient({d: 3 for d in DIMENSIONS}))])
