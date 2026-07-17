# IdeaGraph PDF User Guide — Generation Prompt

Paste this prompt into a code-capable LLM (Claude with code interpreter,
ChatGPT with Code Interpreter, or a local agent with `matplotlib` +
`reportlab` available). The LLM will produce `ideagraph_user_guide.pdf`
in the working directory. The prompt is self-contained — no extra
context required.

---

## SYSTEM ROLE

You are a senior technical writer + information designer producing a
polished PDF user guide for **IdeaGraph**, a Streamlit research-idea
generation app for PhD students and research groups. The guide must be:

- Production-quality (clean layout, generous whitespace, consistent
  typography, every page numbered)
- 30–50 pages long
- Written in plain English, not marketing copy. Each feature gets *what
  it does*, *when to use it*, *how to read its output*, and one concrete
  worked example
- Self-contained — readable without prior knowledge of the codebase
- Bundled with **at least one figure per major section** (≥ 25 figures
  total) generated programmatically with `matplotlib` — do NOT rely on
  screenshots. Figures must be original diagrams that explain mechanics,
  not decorative

## TASK

Generate `ideagraph_user_guide.pdf` in the current working directory.
Use this toolchain (install if missing):

- `matplotlib` for every figure (PNG, 150 dpi minimum)
- `reportlab` for PDF assembly (`SimpleDocTemplate` + flowables)
- Standard library only otherwise

Each figure should be saved to `./figures/fig_<NN>_<slug>.png` and
embedded into the PDF inline at the relevant section. Keep the
`figures/` directory after the PDF is built — operators may reuse the
PNGs in slides.

## OUTPUT STRUCTURE (use exactly these top-level sections)

1. **Cover & Front Matter** — title page, version, contact, table of
   contents.
2. **What IdeaGraph Is** — one-page summary, target user, mental model.
   Figure: high-level data-flow diagram (topic → seed queries → DAG
   construction → QD archive → debate → output).
3. **Quick Start** — minimal end-to-end run (5 steps). Figure: numbered
   screenshot-style mock of the sidebar with annotations.
4. **The Sidebar** — every control:
   - LLM Provider selector (7 providers: DeepSeek, OpenAI, Groq, Gemini,
     Azure, Anthropic Claude, Kimi/Moonshot). Cost-rate table.
   - Topic + budget + iterations sliders
   - Presets (built-in and user-saved)
   - Debate toggle, creativity slider, time/risk knobs
   - Figure: annotated mock sidebar
   - Figure: bar chart of per-provider $/M-token rates
5. **Main Tabs** — one subsection per tab, in this order:
   Scientist · Analytics · Archive · Ideas · Regenerate · Novelty Lab ·
   Simulate · Exec Loop · Reviewer Lens · Provenance · Compare ·
   Mashup · Trends · Collab · Debate · Papers · History · Recommend ·
   Cross-Domain · DAG · Proposal · Evolution · Community · Log.
   Each subsection: 1 paragraph of purpose + 1 paragraph of mechanics +
   1 worked example. Add a figure to ≥ 8 of these tabs (your pick — pick
   the ones with the most non-obvious mechanics).
6. **The Novelty Lab — All 18 Modes** — this is the centerpiece. One
   page per mode. For each mode include:
   - One-line summary
   - Strategy code (single letter)
   - Two-phase description (extraction / generation)
   - Worked example with a placeholder topic
   - Diagram figure showing the mode's mechanism
   The 18 modes in order:

   | # | Mode key | Strategy | One-line essence |
   |---|----------|---------|------------------|
   | 1 | adversarial | N | attack & revise one idea |
   | 2 | contradiction | C | TRIZ-style contradictions |
   | 3 | ensemble | E | parallel cross-provider diversity |
   | 4 | constraint | K | multi-constraint satisfaction |
   | 5 | future_back | U | imagine 2040 → back-propagate |
   | 6 | frontier | X | embedding-gradient frontier |
   | 7 | genetic | G | crossover + mutate |
   | 8 | heretic | H | falsify field consensus |
   | 9 | persona | P | 8-persona swap |
   | 10 | counterfactual | L | imagined abandoned literature |
   | 11 | analogy | M | cross-domain morphism |
   | 12 | failure_mode | Y | antibody-by-design |
   | 13 | extremum | T | pin one axis to its limit |
   | 14 | inversion | I | Jeopardy: answer → question |
   | 15 | null_result | Z | pre-registered clean negative |
   | 16 | underserved_cohort | W | research anchored to a who |
   | 17 | composable_primitive | D | design the building block |
   | 18 | stakeholder_pareto | S | measurable win per stakeholder |

7. **Admin Dashboard** — 4 sub-tabs:
   - Platform Stats — what each metric means
   - Pipeline Simulator — what the simulator demonstrates
   - Population (Federation) — federated MAP-Elites mechanism
   - LLM Provider — runtime switching, persist-to-.env semantics
   Figure: federated MAP-Elites mechanism (broadcast → census →
   saturation-penalty → biased cell selection).
   Figure: provider-switch state machine.
8. **Strategy-Code Reference** — alphabetical table of every
   `source_strategy` letter currently in use (A, B, C, D, E, F, G, H,
   I, K, L, M, N, P, R, S, T, U, W, X, Y, Z) with the module that
   emits it and one-line meaning. Figure: a 5×5 grid colour-coded by
   broad family (offense / defense / structural / cohort / regime).
9. **Configuration & .env Reference** — table of every environment
   variable read in `config.py` with default and brief purpose.
10. **Glossary** — Idea, QD archive, methodology type, novelty level,
    coverage, Div-Pair, homogenization, equivalence margin,
    pre-registration, adoption proxy, Pareto front.
11. **Troubleshooting** — top 10 issues + resolutions (port 8510
    busy, missing API key, kimi-k2 temperature coercion, Anthropic
    proxy 502, etc.).
12. **Appendix A — Architecture Diagram** — full figure of the
    pipeline modules (`agents/`, `models/idea.py`, the optimization
    layers, the 18 novelty modules, the admin dashboard).

## FIGURE REQUIREMENTS

Build every figure with `matplotlib`. Use this design system uniformly:

- Palette: `#0c4a6e` (deep blue, headings), `#0ea5e9` (sky, accent),
  `#10b981` (green, success), `#f59e0b` (amber, warning), `#ef4444`
  (red, failure), `#64748b` (slate, secondary)
- Font: use `matplotlib`'s default sans-serif; titles 14pt bold, axis
  labels 11pt
- No 3D plots, no gridlines on flow diagrams, no decorative emoji in
  figures
- Every figure has a numbered caption underneath (`Figure NN. …`)

Required figures (≥ 25 total) — minimum list:

1. High-level data flow (topic → ideas)
2. Sidebar annotated mock
3. Provider $/M-token bar chart
4. QD archive grid (7 methodology rows × 3 novelty cols) with
   illustrative occupancy
5. Pipeline state machine
6. Debate tournament bracket
7. Federated MAP-Elites mechanism flow
8. Strategy-code family grid
9. One mechanism diagram per Novelty Lab mode (×18 — minimum
   acceptable: a flow with two boxes labelled "Phase 1" and "Phase 2"
   and the artefact each emits, with the mode-specific catalog listed
   on the side)
10. Architecture diagram (appendix)

Total: 8 + 18 = 26 figures minimum. Aim for 30 for headroom.

## STYLE GUIDE

- Body text: serif (reportlab default), 10pt, 1.4× leading
- Headings: sans-serif, sized 22 / 16 / 13 / 11
- Code/identifiers inline use `monospace`
- Tables: thin rules, no zebra striping, header row bold
- Margins: 0.85 inch all around
- Page numbers bottom-center
- Running header: "IdeaGraph User Guide — v{VERSION}"
- Inject `VERSION = "2026-05"` at the top of your script and refer to
  it everywhere; do not hardcode dates in text

## CONSTRAINTS

- Do NOT invent features that are not in this prompt. If something is
  unclear, write a placeholder and flag it with a "⚠️ EDITOR NOTE:" line
  the operator can hand-edit. Hallucinating undocumented behaviour is
  the single worst failure mode for this artifact.
- Do NOT include screenshots of the live app. All figures must be
  generated by your script. Mock-ups of the UI must be obviously
  diagrammatic (boxes + labels), never claim to be a screenshot.
- Do NOT use marketing language ("revolutionary", "game-changing",
  "next-gen"). Plain technical English only.
- Treat the 18 Novelty Lab modes as the most important part of the
  guide. If you run out of time/tokens, prioritize completing all 18
  mode pages over polishing later sections.

## VALIDATION CHECKLIST (run before declaring done)

- [ ] `ideagraph_user_guide.pdf` exists and opens
- [ ] PDF has a table of contents listing every section
- [ ] All 18 novelty modes have their own dedicated page or pages
- [ ] ≥ 25 figures embedded, every figure has a numbered caption
- [ ] Every figure file in `./figures/` is also embedded in the PDF
- [ ] Page numbers run continuously
- [ ] No marketing copy, no fabricated features
- [ ] Strategy-code table covers all 21 letters in use
- [ ] Glossary defines every term used 3+ times in the body

## DELIVERABLES

Write a Python script `build_guide.py` that, when run, produces:

1. `ideagraph_user_guide.pdf` (the artifact)
2. `figures/` (directory with all generated PNGs)
3. `build_guide.log` (one-line summary per generated figure + final
   PDF page count)

Execute the script. Verify the validation checklist. Report the final
page count, figure count, and any "⚠️ EDITOR NOTE:" lines you inserted.
