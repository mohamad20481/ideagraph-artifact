"""Tests for corpus_novelty — operationalized novelty against a reference corpus (strategy Q)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch
from typing import List

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import corpus_novelty as cn
from corpus_novelty import (
    CORPUS_LIMITS_DISCLAIMER,
    CorpusEntry,
    NoveltyAssessment,
    ReferenceCorpus,
    assess_idea_novelty,
    corpus_anchored_batch,
    cosine_similarity,
    text_vector,
    tokenize,
)
from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


# ── tokenize ────────────────────────────────────────────────────────────────

def test_tokenize_empty_returns_empty():
    assert tokenize("") == []
    assert tokenize(None) == []  # type: ignore[arg-type]


def test_tokenize_filters_stopwords_and_short_words():
    out = tokenize("The transformer is a new attention method.")
    assert "the" not in out
    assert "is" not in out
    assert "transformer" in out
    assert "attention" in out
    # 'a' is too short and stopword
    assert "a" not in out


def test_tokenize_lowercases_and_strips_punctuation():
    out = tokenize("BERT-Large achieves SOTA on GLUE!")
    assert "bert-large" in out or "bert" in out
    assert "glue" in out
    # No punctuation tokens
    assert "!" not in out
    assert "," not in out


def test_tokenize_filters_research_glue_words():
    """Common research-paper filler should be filtered."""
    out = tokenize("We propose a novel method that shows results.")
    assert "propose" not in out
    assert "novel" not in out
    assert "method" not in out
    assert "shows" not in out


# ── text_vector / cosine_similarity ─────────────────────────────────────────

def test_text_vector_empty_returns_empty():
    assert text_vector("") == {}


def test_text_vector_unit_normalized():
    v = text_vector("transformer attention attention attention")
    norm_sq = sum(x * x for x in v.values())
    assert abs(norm_sq - 1.0) < 1e-9


def test_cosine_self_similarity_is_one():
    v = text_vector("graph neural networks for protein folding")
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal_disjoint_vocabularies():
    v1 = text_vector("transformer attention scaling")
    v2 = text_vector("kidney dialysis pediatric")
    assert cosine_similarity(v1, v2) == 0.0


def test_cosine_partial_overlap_is_between_zero_and_one():
    v1 = text_vector("attention transformer")
    v2 = text_vector("attention recurrent")
    sim = cosine_similarity(v1, v2)
    assert 0.0 < sim < 1.0


def test_cosine_empty_vector_returns_zero():
    assert cosine_similarity({}, text_vector("hello")) == 0.0
    assert cosine_similarity(text_vector("hello"), {}) == 0.0


# ── CorpusEntry ─────────────────────────────────────────────────────────────

def test_corpus_entry_text_with_abstract():
    e = CorpusEntry(title="Attention is all you need", abstract="A new model.")
    assert "Attention" in e.text()
    assert "new model" in e.text()


def test_corpus_entry_text_without_abstract():
    e = CorpusEntry(title="Just a title")
    assert e.text() == "Just a title"


def test_corpus_entry_excerpt_truncates_long_abstracts():
    long = "x" * 500
    e = CorpusEntry(title="t", abstract=long, excerpt_max=100)
    ex = e.excerpt()
    assert len(ex) <= 105
    assert ex.endswith("…")


def test_corpus_entry_excerpt_no_truncation_when_short():
    e = CorpusEntry(title="t", abstract="short", excerpt_max=100)
    assert e.excerpt() == "short"


# ── ReferenceCorpus ─────────────────────────────────────────────────────────

def test_corpus_empty_has_len_zero():
    c = ReferenceCorpus()
    assert len(c) == 0
    assert c.entries == []


def test_corpus_add_validates_title():
    c = ReferenceCorpus()
    with pytest.raises(ValueError):
        c.add(title="")
    with pytest.raises(ValueError):
        c.add(title="   ")


def test_corpus_add_caches_vector():
    c = ReferenceCorpus()
    c.add(title="Transformer attention", abstract="Self-attention layers.")
    assert len(c) == 1
    e = c.entries[0]
    assert e._vec is not None
    assert len(e._vec) > 0


def test_corpus_nearest_neighbor_finds_closest():
    c = ReferenceCorpus.from_seed_texts([
        "Transformer attention. Self-attention layers.",
        "Kidney dialysis. Renal replacement therapy.",
    ])
    nn, sim = c.nearest_neighbor(
        "Attention-based transformer architectures"
    )
    assert nn is not None
    assert "Transformer" in nn.title
    assert sim > 0.0


def test_corpus_nearest_neighbor_empty_corpus():
    c = ReferenceCorpus()
    nn, sim = c.nearest_neighbor("anything")
    assert nn is None
    assert sim == 0.0


def test_corpus_nearest_neighbor_empty_query_text():
    """If the query tokenizes to empty (only stopwords), no match."""
    c = ReferenceCorpus.from_seed_texts(["Some title. Some abstract."])
    nn, sim = c.nearest_neighbor("the a is of")
    assert nn is None
    assert sim == 0.0


def test_corpus_novelty_score_is_one_minus_similarity():
    c = ReferenceCorpus.from_seed_texts([
        "Transformer attention. Self-attention layers.",
    ])
    # Same text → similarity ~1 → novelty ~0
    nov_same = c.novelty_score("Transformer attention. Self-attention layers.")
    assert nov_same < 0.1
    # Orthogonal text → similarity ~0 → novelty ~1
    nov_far = c.novelty_score("Kidney dialysis pediatric care")
    assert nov_far > 0.9


def test_corpus_from_seed_texts_separates_title_and_abstract():
    c = ReferenceCorpus.from_seed_texts([
        "My Title. The abstract here.",
        "No-abstract title only",
    ])
    assert len(c) == 2
    titles = [e.title for e in c.entries]
    assert "My Title" in titles
    assert "No-abstract title only" in titles
    # The first one should have an abstract.
    first = next(e for e in c.entries if e.title == "My Title")
    assert "abstract here" in first.abstract


def test_corpus_from_seed_texts_drops_blank_lines():
    c = ReferenceCorpus.from_seed_texts(["", "  ", "Real title"])
    assert len(c) == 1


def test_corpus_from_archive_uses_title_and_method():
    archive = [
        {"title": "Idea A", "motivation": "mot A", "method": "method A"},
        {"title": "Idea B", "motivation": "mot B", "method": "method B"},
        {"title": "", "method": "skip me"},  # no title → dropped
    ]
    c = ReferenceCorpus.from_archive(archive)
    assert len(c) == 2
    e = c.entries[0]
    assert e.source == "archive"
    assert "mot A" in e.abstract
    assert "method A" in e.abstract


def test_corpus_custom_embedder():
    """Pluggable embedder swaps in cleanly."""
    def dummy_embedder(text: str) -> dict:
        return {"fixed_token": 1.0}

    c = ReferenceCorpus(embedder=dummy_embedder)
    c.add(title="anything", abstract="anything")
    # Self-similarity should be 1.0 because all vectors are identical.
    assert c.novelty_score("totally different text") < 0.01


# ── assess_idea_novelty ─────────────────────────────────────────────────────

_SAMPLE_IDEA = {
    "title": "Transformer attention scaling",
    "motivation": "Self-attention is O(n^2).",
    "method": "Linear attention via kernel approximation.",
    "hypothesis": "Linear attention matches accuracy at lower cost.",
    "methodology_type": METHODOLOGY_TYPES[0],
    "novelty_level": NOVELTY_LEVELS[1],
}


def test_assess_idea_novelty_invalid_idea_raises():
    c = ReferenceCorpus()
    with pytest.raises(ValueError):
        assess_idea_novelty({}, c)
    with pytest.raises(ValueError):
        assess_idea_novelty(None, c)  # type: ignore[arg-type]


def test_assess_idea_novelty_invalid_corpus_raises():
    with pytest.raises(ValueError):
        assess_idea_novelty(_SAMPLE_IDEA, "not a corpus")  # type: ignore[arg-type]


def test_assess_idea_novelty_empty_corpus_returns_max_novelty():
    c = ReferenceCorpus()
    a = assess_idea_novelty(_SAMPLE_IDEA, c)
    assert a.score == 1.0
    assert a.nearest_title == "(corpus empty)"
    assert a.corpus_size == 0


def test_assess_idea_novelty_against_orthogonal_corpus_high_score():
    c = ReferenceCorpus.from_seed_texts([
        "Kidney dialysis. Renal replacement therapy.",
        "Mediterranean diet. Olive oil cardiovascular benefits.",
    ])
    a = assess_idea_novelty(_SAMPLE_IDEA, c)
    assert a.score > 0.9
    assert a.corpus_size == 2
    assert a.nearest_title in {"Kidney dialysis", "Mediterranean diet"}


def test_assess_idea_novelty_against_similar_corpus_low_score():
    c = ReferenceCorpus.from_seed_texts([
        "Transformer attention scaling. Self-attention is O(n^2). "
        "Linear attention reduces cost.",
    ])
    a = assess_idea_novelty(_SAMPLE_IDEA, c)
    assert a.score < 0.5
    assert "Transformer" in a.nearest_title
    assert a.nearest_similarity > 0.5


def test_assess_idea_novelty_attaches_nearest_excerpt():
    c = ReferenceCorpus.from_seed_texts([
        "Attention is all you need. Transformer model using only attention.",
    ])
    a = assess_idea_novelty(_SAMPLE_IDEA, c)
    assert a.nearest_excerpt
    assert "Transformer" in a.nearest_excerpt or "attention" in a.nearest_excerpt


def test_assessment_to_dict_roundtrip():
    a = NoveltyAssessment(
        score=0.5, nearest_similarity=0.5, nearest_title="t",
        nearest_source="s", nearest_year=2023, nearest_excerpt="x",
        corpus_size=10,
    )
    d = a.to_dict()
    assert d["score"] == 0.5
    assert d["nearest_year"] == 2023
    assert d["corpus_size"] == 10


# ── corpus_anchored_batch ───────────────────────────────────────────────────

def _make_idea(title: str, method: str = "m", hypothesis: str = "h",
                  meth: str = METHODOLOGY_TYPES[0],
                  nov: str = NOVELTY_LEVELS[1],
                  strategy: str = "P") -> Idea:
    return Idea(
        title=title, motivation="mot", method=method, hypothesis=hypothesis,
        resources="r", expected_outcome="e", risk_assessment="ra",
        source_strategy=strategy,
        methodology_type=meth, novelty_level=nov,
    )


def test_batch_empty_topic_raises():
    c = ReferenceCorpus()
    with pytest.raises(ValueError):
        corpus_anchored_batch("", c)


def test_batch_invalid_corpus_raises():
    with pytest.raises(ValueError):
        corpus_anchored_batch("topic", "not a corpus")  # type: ignore[arg-type]


def test_batch_invalid_min_novelty_raises():
    c = ReferenceCorpus()
    with pytest.raises(ValueError):
        corpus_anchored_batch("topic", c, min_novelty=2.0)
    with pytest.raises(ValueError):
        corpus_anchored_batch("topic", c, min_novelty=-0.1)


def test_batch_invalid_generator_raises():
    c = ReferenceCorpus()
    with pytest.raises(ValueError):
        corpus_anchored_batch("topic", c, generators=["nope"])


def test_batch_zero_keep_returns_empty():
    c = ReferenceCorpus()
    assert corpus_anchored_batch("topic", c, keep_top=0) == []


def test_batch_no_candidates_returns_empty():
    """If _gather_candidates yields nothing, batch returns []."""
    c = ReferenceCorpus.from_seed_texts(["seed entry"])
    with patch("corpus_novelty._gather_candidates", return_value=[]):
        out = corpus_anchored_batch(
            "topic", c, generators=["persona"],
            n_per_generator=2, keep_top=5,
        )
    assert out == []


def test_batch_scores_sorts_and_stamps_strategy_Q():
    """End-to-end happy path: candidates get scored, sorted by novelty
    desc, and stamped with strategy='Q'."""
    corpus = ReferenceCorpus.from_seed_texts([
        "Transformer attention scaling. Linear attention via kernels.",
    ])
    fake_candidates = [
        # Highly novel (orthogonal to corpus).
        _make_idea(
            "Kidney dialysis hemoglobin",
            method="renal replacement therapy",
            hypothesis="dialysis improves outcomes",
            meth=METHODOLOGY_TYPES[1], nov=NOVELTY_LEVELS[2],
            strategy="P",
        ),
        # Low novelty (matches corpus).
        _make_idea(
            "Transformer attention scaling",
            method="Linear attention via kernel approximation",
            hypothesis="Linear attention matches accuracy",
            meth=METHODOLOGY_TYPES[2], nov=NOVELTY_LEVELS[0],
            strategy="M",
        ),
    ]
    with patch("corpus_novelty._gather_candidates",
                return_value=fake_candidates):
        out = corpus_anchored_batch(
            "topic", corpus, n_per_generator=1, keep_top=5,
            map_elites=False,
        )

    assert len(out) == 2
    # Both stamped Q with upstream preserved.
    assert all(i.source_strategy == "Q" for i in out)
    assert out[0].execution_meta["upstream_strategy"] == "P"
    assert out[1].execution_meta["upstream_strategy"] == "M"
    # Sorted by novelty desc → kidney first.
    assert "Kidney" in out[0].title
    # Each carries the corpus_novelty assessment.
    nov_top = out[0].execution_meta["corpus_novelty"]
    nov_bot = out[1].execution_meta["corpus_novelty"]
    assert nov_top["score"] > nov_bot["score"]
    # Corpus limits disclaimer attached to every idea.
    assert out[0].execution_meta["corpus_limits"] == CORPUS_LIMITS_DISCLAIMER
    assert out[1].execution_meta["corpus_limits"] == CORPUS_LIMITS_DISCLAIMER
    # regen_mode + topic stamped.
    assert out[0].execution_meta["regen_mode"] == "corpus_anchored"
    assert out[0].execution_meta["topic"] == "topic"


def test_batch_min_novelty_floor_filters():
    corpus = ReferenceCorpus.from_seed_texts([
        "Transformer attention scaling. Linear attention via kernels.",
    ])
    fake = [
        _make_idea("Kidney dialysis hemoglobin"),                         # high novelty
        _make_idea("Transformer attention scaling kernel approximation"),  # low
    ]
    with patch("corpus_novelty._gather_candidates", return_value=fake):
        out = corpus_anchored_batch(
            "topic", corpus, n_per_generator=1, keep_top=5,
            min_novelty=0.7, map_elites=False,
        )
    # Only the kidney idea clears the 0.7 floor.
    assert len(out) == 1
    assert "Kidney" in out[0].title


def test_batch_map_elites_dedup_keeps_most_novel_per_cell():
    """Two ideas in the same cell — only the more novel one survives."""
    corpus = ReferenceCorpus.from_seed_texts([
        "Transformer attention scaling. Linear attention via kernels.",
    ])
    meth = METHODOLOGY_TYPES[0]
    nov = NOVELTY_LEVELS[1]
    fake = [
        _make_idea("Kidney dialysis", meth=meth, nov=nov),
        _make_idea("Transformer attention", meth=meth, nov=nov),
    ]
    with patch("corpus_novelty._gather_candidates", return_value=fake):
        out = corpus_anchored_batch(
            "topic", corpus, n_per_generator=1, keep_top=5,
            map_elites=True,
        )
    assert len(out) == 1
    assert "Kidney" in out[0].title


def test_batch_keep_top_truncates():
    corpus = ReferenceCorpus.from_seed_texts(["unrelated content seed"])
    fake = [
        _make_idea(f"Idea {idx}", meth=METHODOLOGY_TYPES[idx % 7],
                       nov=NOVELTY_LEVELS[idx % 3])
        for idx in range(10)
    ]
    with patch("corpus_novelty._gather_candidates", return_value=fake):
        out = corpus_anchored_batch(
            "topic", corpus, n_per_generator=1, keep_top=3,
            map_elites=False,
        )
    assert len(out) == 3


def test_batch_default_generators_used_when_none_passed():
    """Verify the default generator list is wired correctly."""
    c = ReferenceCorpus.from_seed_texts(["seed"])
    called_with = {}
    def _capture_gather(topic, generators, n_per, claude):
        called_with["generators"] = list(generators)
        return []
    with patch("corpus_novelty._gather_candidates",
                side_effect=_capture_gather):
        corpus_anchored_batch("topic", c, generators=None)
    assert called_with["generators"] == ["persona", "analogy", "heretic"]


# ── Strategy code Q ─────────────────────────────────────────────────────────

def test_strategy_code_Q_distinct_from_existing():
    existing = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L",
                "M", "N", "P", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"}
    assert "Q" not in existing


def test_strategy_Q_present_in_module():
    text = (ROOT / "corpus_novelty.py").read_text(encoding="utf-8")
    assert 'source_strategy = "Q"' in text


# ── App wiring smoke checks ─────────────────────────────────────────────────

def test_app_novelty_lab_has_19_modes():
    app_text = (ROOT / "app.py").read_text(encoding="utf-8")
    idx = app_text.find("_novelty_mode = st.radio")
    assert idx >= 0
    snippet = app_text[idx:idx + 4000]
    expected = [
        "adversarial", "contradiction", "ensemble", "constraint",
        "future_back", "frontier", "genetic", "heretic", "persona",
        "counterfactual", "analogy", "failure_mode", "extremum",
        "inversion", "null_result", "underserved_cohort",
        "composable_primitive", "stakeholder_pareto", "corpus_anchored",
    ]
    assert len(expected) == 19
    for mode in expected:
        assert f'"{mode}"' in snippet, f"mode {mode!r} missing"


def test_app_imports_corpus_novelty():
    app_text = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "from corpus_novelty import" in app_text
    assert "CORPUS_LIMITS_DISCLAIMER" in app_text
    assert "corpus_anchored_batch" in app_text


def test_corpus_limits_disclaimer_non_empty_and_honest():
    """Disclaimer must explicitly say it does NOT prove no one ever
    suggested the idea."""
    assert len(CORPUS_LIMITS_DISCLAIMER) > 100
    assert "NOT" in CORPUS_LIMITS_DISCLAIMER or "does not" in \
        CORPUS_LIMITS_DISCLAIMER.lower()


# ── Custom embedder integration ────────────────────────────────────────────

def test_custom_embedder_propagates_to_scoring():
    def char_embedder(text: str) -> dict:
        # Character-frequency over alpha-only chars.
        out = {}
        for ch in (text or "").lower():
            if ch.isalpha():
                out[ch] = out.get(ch, 0.0) + 1.0
        if not out:
            return {}
        import math
        norm = math.sqrt(sum(v * v for v in out.values()))
        return {k: v / norm for k, v in out.items()}

    c = ReferenceCorpus(embedder=char_embedder)
    c.add(title="abcdefg")
    # Word-vector mode would give 0 here; char-level should not.
    nn, sim = c.nearest_neighbor("gfedcba")
    assert sim > 0.5  # same char distribution
