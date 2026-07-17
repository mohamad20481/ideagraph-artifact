# IdeaGraph Evaluation Harness

**Added 2026-07, after ARR reviews of submission 599.** Reviewers correctly
noted that the paper's metrics were not formally defined and that this
repository did not contain the evaluation code. This package is the fix —
built **after** those reviews.

## Read this first (honesty contract)

1. **This is a fresh implementation.** The definitions below were fixed at
   the time this package was written. Running this harness produces **new
   numbers** that **supersede any previously reported results**. No part of
   this package was tuned to reproduce any earlier table, and its outputs
   should not be expected to match them.
2. **Benchmarks are data, not code.** `crossval.py` provides the schema,
   validation, and blinding for a CrossVal-style benchmark. The benchmark
   itself must be produced by real work: generated entries from actual
   pipeline runs, published entries extracted from real papers with
   provenance (paper title, venue, year) and **human verification**. The
   loader refuses published entries without provenance.
3. **Human panels are people.** This package contains no human ratings and
   cannot create them. If a paper reports a human panel, the underlying
   rating data must exist and be released.
4. **Evaluation fails loudly.** Every function raises `EvaluationError` on
   missing prerequisites instead of silently scoring 0. A run is complete or
   it is failed — never silently partial.

## Canonical metric definitions

Embeddings: `e(x)` = L2-normalized sentence embedding of the idea's canonical
text (`metrics.idea_to_text`: title, motivation, hypothesis, method/
method_sketch, dataset, metrics, baselines, expected_outcome — non-empty
fields joined in that order). Default model: the project bi-encoder
(`sentence-transformers/all-MiniLM-L6-v2`); the model used must be reported.

**Pairwise dissimilarity** — `d(i,j) = 1 − max(0, cos(e_i, e_j))` ∈ [0,1].

**Diversity Index (DI)** — per topic `t` with idea set `I_t` (|I_t| ≥ 2):

    DI_t = (2 / (|I_t|·(|I_t|−1))) · Σ_{i<j} d(i,j)        DI = mean_t DI_t

**Div-Pair** — the identical formula applied to a flat pool of ideas.

**Reference Alignment Score (RAS)** — per topic `t` with reference text `r_t`
(reference paper's contribution statement; title+abstract acceptable and must
be stated):

    RAS_t = agg_{i∈I_t} cos(e_i, e(r_t)),  agg ∈ {max (default), mean}
    RAS   = mean_t max(0, RAS_t)

The `agg` used must be reported. `max` = "did the best idea recover the
reference contribution"; `mean` = portfolio-level alignment.

**Judge scores** — six dimensions (novelty, feasibility, clarity,
significance, excitement, overall), anchored 1–5 rubric (`judges.RUBRIC`),
temperature 0.0, blind 7-field tuples only. Per-idea score = mean across the
judge panel. Unit mapping: `u = (s − 1) / 4`.

**Nov-A / Feas-A** — mean over ideas of the judge-averaged unit novelty /
feasibility.

**Novelty–Feasibility Tradeoff (NFT)** — per idea, the harmonic mean:

    NFT_i = 2·nov_i·feas_i / (nov_i + feas_i)   (0 if both are 0)
    NFT   = mean_i NFT_i

Harmonic mean is deliberate: an idea maximally novel but infeasible scores
low. Per-idea-then-average is deliberate: a portfolio cannot score well by
mixing purely-novel with purely-feasible ideas.

**Non-inferiority (TOST)** — Welch two-sample; margin δ > 0, higher = better:
`H0: μ_gen − μ_pub ≤ −δ` vs `H1: μ_gen − μ_pub > −δ`;
`t = (d + δ)/SE`, one-sided `p = P(T_df > t)`; established iff `p < α`.
Full equivalence TOST and (1−2α) CIs in `tost.py`. Discriminability =
forced-choice accuracy with Clopper–Pearson CI + exact binomial test.

## Layout

    evaluation/
      metrics.py     RAS, DI/Div-Pair, NFT (+ canonical text & embedding)
      tost.py        Welch non-inferiority, equivalence TOST, Clopper–Pearson
      crossval.py    7-field tuple schema, JSONL IO, validation, blinding
      judges.py      blind LLM judge panel (anchored rubric, complete-or-fail)
      run_eval.py    CLI orchestrator (see --help)
    tests/test_evaluation.py   deterministic tests (stub embedder/judges)

## What a real run requires

1. Generated ideas: actual pipeline outputs saved per topic.
2. Published entries: human-verified extractions with provenance.
3. A judge panel: ≥1 configured provider (3 distinct providers recommended;
   report the exact panel).
4. `python -m evaluation.run_eval --help` for the commands.

Steps 1–3 cost real API budget and real human time. There is no shortcut.
