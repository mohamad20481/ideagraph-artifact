"""
config.py - Central configuration for IdeaGraph
Loads .env and exposes all settings as module-level constants.
"""

import os
from typing import Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (same directory as this file)
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(_ENV_PATH)

# ── Provider selection ────────────────────────────────────────────────────────
PROVIDER: str = os.getenv("IDEAGRAPH_PROVIDER", "deepseek").lower()

# ── Feature toggles (operator-settable from the admin dashboard) ────────────
# Corpus-anchored novelty (Novelty Lab mode `corpus_anchored`, strategy `Q`).
# Default True so the feature is discoverable; admins can disable per
# deployment to hide the mode from the radio.
ENABLE_CORPUS_ANCHORED_NOVELTY: bool = os.getenv(
    "IDEAGRAPH_CORPUS_ANCHORED_NOVELTY", "true",
).lower() == "true"

# Auto-score arXiv novelty after every "Run Automated Scientist" pass.
# When ON (default), the pipeline runner fetches the topic's live arXiv
# corpus once and stamps execution_meta["arxiv_novelty"] onto each idea
# so the 📡 arXiv Novelty score/sort is populated without a manual click.
# Best-effort: a rate-limit / network failure never aborts the run.
# Kill-switch for deployments that want to avoid the extra arXiv call.
AUTO_ARXIV_NOVELTY: bool = os.getenv(
    "IDEAGRAPH_AUTO_ARXIV_NOVELTY", "true",
).lower() == "true"

# Corpus size (max arXiv papers) for the auto-scoring pass above.
# Matches the manual panel's default of 30. 1 arXiv call, disk-cached.
AUTO_ARXIV_MAX_PAPERS: int = int(os.getenv("IDEAGRAPH_AUTO_ARXIV_MAX_PAPERS", "30"))

# Auto-score *blended* multi-corpus novelty after every run too.
# Blends arXiv + Semantic Scholar + OpenAlex (the legitimate, ToS-clean
# stand-in for ResearchGate, which has no usable API) via the weights in
# multi_corpus_novelty.DEFAULT_WEIGHTS, stamping
# execution_meta["blended_novelty"] onto each idea. Best-effort: any
# source failing (rate-limit / network) just drops its weight; the run
# is never aborted. Kill-switch for deployments wanting fewer API calls.
AUTO_BLENDED_NOVELTY: bool = os.getenv(
    "IDEAGRAPH_AUTO_BLENDED_NOVELTY", "true",
).lower() == "true"

# Corpus size per source for the blended auto-scoring pass.
AUTO_BLENDED_MAX_PAPERS: int = int(os.getenv("IDEAGRAPH_AUTO_BLENDED_MAX_PAPERS", "30"))

# OpenAlex polite-pool contact. Setting an email routes requests to
# OpenAlex's faster, more reliable pool. Optional but recommended.
OPENALEX_MAILTO: str = os.getenv("OPENALEX_MAILTO", "")

# Crossref polite-pool contact (falls back to OPENALEX_MAILTO if unset).
# Both arXiv/S2/OpenAlex/Crossref/Europe PMC feed the blended novelty
# score; Europe PMC needs no contact email.
CROSSREF_MAILTO: str = os.getenv("CROSSREF_MAILTO", "")

# Semantic (dense-embedding) novelty — the "advanced" tier. Unlike every
# other scorer (which is lexical TF-IDF), this measures 1 − max cosine in
# sentence-embedding space, catching conceptual novelty/paraphrase that
# word-overlap misses. Auto-scored after each run (best-effort: if the
# embedding model can't load, the run is unaffected). Reuses the cached
# arXiv corpus — no extra network call beyond the arXiv auto-score.
AUTO_SEMANTIC_NOVELTY: bool = os.getenv(
    "IDEAGRAPH_AUTO_SEMANTIC_NOVELTY", "true",
).lower() == "true"

# sentence-transformers model for semantic novelty. Default all-MiniLM-L6-v2
# (~80MB, fast, downloaded once from HuggingFace and disk-cached).
SEMANTIC_NOVELTY_MODEL: str = os.getenv(
    "IDEAGRAPH_SEMANTIC_NOVELTY_MODEL", "sentence-transformers/all-MiniLM-L6-v2",
)

# Corpus size (max arXiv papers) for the semantic auto-scoring pass.
AUTO_SEMANTIC_MAX_PAPERS: int = int(os.getenv("IDEAGRAPH_AUTO_SEMANTIC_MAX_PAPERS", "30"))

# Cross-encoder RERANKED novelty — the top tier (more accurate than the
# bi-encoder semantic score). Two-stage retrieve-then-rerank: bi-encoder
# shortlists top-K papers, cross-encoder rescoring those pairs jointly.
# Auto-scored after each run (best-effort; reuses the cached arXiv corpus
# + shared bi-encoder). Heavier than the other tiers (one transformer
# forward pass per (idea, candidate) pair), so it has its own kill-switch.
AUTO_RERANKED_NOVELTY: bool = os.getenv(
    "IDEAGRAPH_AUTO_RERANKED_NOVELTY", "true",
).lower() == "true"

# Cross-encoder model for reranked novelty (~80MB, downloaded once).
CROSSENCODER_NOVELTY_MODEL: str = os.getenv(
    "IDEAGRAPH_CROSSENCODER_NOVELTY_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

# Corpus size + shortlist depth for the reranked auto-scoring pass.
AUTO_RERANKED_MAX_PAPERS: int = int(os.getenv("IDEAGRAPH_AUTO_RERANKED_MAX_PAPERS", "30"))
RERANKED_TOP_K: int = int(os.getenv("IDEAGRAPH_RERANKED_TOP_K", "15"))

# Mahalanobis (whitened-embedding) DISTRIBUTIONAL novelty. Models the
# corpus as a Gaussian in embedding space and measures distance to the
# whole distribution (anisotropy-aware), not just the nearest paper.
# Reuses the shared bi-encoder + cached arXiv corpus; adds only a small
# Ledoit-Wolf covariance fit. Best-effort, gated, runs after the others.
AUTO_MAHALANOBIS_NOVELTY: bool = os.getenv(
    "IDEAGRAPH_AUTO_MAHALANOBIS_NOVELTY", "true",
).lower() == "true"

# Corpus size (max arXiv papers) for the Mahalanobis auto-scoring pass.
AUTO_MAHALANOBIS_MAX_PAPERS: int = int(os.getenv("IDEAGRAPH_AUTO_MAHALANOBIS_MAX_PAPERS", "30"))

# Visual rendering (🎨 Visual Abstract panel on each idea card).
# Uses the Nano Banana / FLUX image API. When OFF, the panel is hidden
# from the Ideas tab and no requests are made even if a key is set.
ENABLE_VISUAL_RENDERING: bool = os.getenv(
    "IDEAGRAPH_VISUAL_RENDERING", "true",
).lower() == "true"

# ── Visual rendering API key ─────────────────────────────────────────────────
# Required for the 🎨 Visual Abstract feature. Two env var names accepted —
# NANO_BANANA_API_KEY is the canonical name from the setup script;
# BFL_API_KEY is the BlackForest Labs convention. Either works.
NANO_BANANA_API_KEY: str = (
    os.getenv("NANO_BANANA_API_KEY", "")
    or os.getenv("BFL_API_KEY", "")
)
# Pluggable model name. Default depends on the provider:
#   flux_bfl       → flux-pro-1.0
#   gemini_imagen  → imagen-3.0-generate-002
# When empty, the renderer falls back to PROVIDER_DEFAULTS in
# ideagraph_image_renderer.
NANO_BANANA_MODEL: str = os.getenv("NANO_BANANA_MODEL", "")

# Endpoint override — swap to Runway / Replicate / a self-hosted FLUX
# by setting this. When empty, the renderer falls back to per-provider
# defaults (e.g. https://api.bfl.ml/v1 for flux_bfl,
# https://generativelanguage.googleapis.com/v1beta for gemini_imagen).
NANO_BANANA_ENDPOINT: str = os.getenv("NANO_BANANA_ENDPOINT", "")

# Which image-generation provider to use. Registered providers:
#   flux_bfl            — BlackForest Labs FLUX (default)
#   gemini_imagen       — Google AI Studio Imagen 3
#   gemini_flash_image  — Google Gemini Flash Image (generateContent)
#   veo                 — Google Veo 3 (video; long-running operation)
#   grok                — xAI Grok image generation (grok-2-image)
NANO_BANANA_PROVIDER: str = os.getenv("NANO_BANANA_PROVIDER", "flux_bfl")

# ── xAI Grok (image generation only) ─────────────────────────────────────────
# Used when NANO_BANANA_PROVIDER=grok. Endpoint is OpenAI-compatible at
# https://api.x.ai/v1/images/generations. xAI does NOT currently expose a
# public video-generation API; for video stick with the `veo` provider.
GROK_API_KEY: str = os.getenv("GROK_API_KEY", "") or os.getenv("XAI_API_KEY", "")

# ── xAI Grok (LLM chat) ──────────────────────────────────────────────────────
# Reuses GROK_API_KEY above (same xAI account / same secret prefix `xai-...`).
# The LLM dispatcher uses this when config.PROVIDER == "xai".
XAI_BASE_URL: str = "https://api.x.ai/v1"

# Curated list of models surfaced when provider == 'xai' (api.x.ai
# endpoint). Includes BOTH xAI's native Grok lineup AND DeepSeek V4
# models that xAI hosts on the same endpoint — your account/tier
# determines which subset is actually callable. If you hit "The
# supported API model names are X or Y" 400 errors, switch to one of
# the models the error message lists.
XAI_KNOWN_MODELS: list = [
    # ── DeepSeek V4 (hosted by xAI; available to most current accounts) ──
    "deepseek-v4-pro",       # full-quality DeepSeek V4 via xAI
    "deepseek-v4-flash",     # faster / cheaper DeepSeek V4 variant
    # ── xAI native Grok (requires Grok-tier access on your xAI account) ──
    "grok-2-latest",         # production alias — auto-routes to current grok-2
    "grok-3",                # newest flagship (beta access required)
    "grok-3-mini",           # smaller / cheaper grok-3 variant
    "grok-2-1212",           # pinned grok-2 (December 2024 snapshot)
    "grok-2",                # generic grok-2
    "grok-vision-beta",      # multimodal beta
    "grok-beta",             # legacy beta
]

# ── API keys & base URLs ──────────────────────────────────────────────────────
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

AZURE_API_KEY: str = os.getenv("AZURE_API_KEY", "")
AZURE_BASE_URL: str = os.getenv("AZURE_BASE_URL", "")

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "") or os.getenv("ANTHROPIC_AUTH_TOKEN", "")

# Kimi (Moonshot AI) — OpenAI-compatible. Use the .ai endpoint by default;
# users inside China can override KIMI_BASE_URL to https://api.moonshot.cn/v1.
KIMI_API_KEY: str = os.getenv("KIMI_API_KEY", "") or os.getenv("MOONSHOT_API_KEY", "")
KIMI_BASE_URL: str = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")

# ── Speed optimizations ────────────────────────────────────────────────────
# Route probes (high-volume, low-creativity) to a cheaper/faster model
# automatically. Saves ~60% wall-clock time on long pipelines.
ENABLE_STAGE_ROUTING: bool = os.getenv("ENABLE_STAGE_ROUTING", "true").lower() == "true"
ENABLE_PROBE_SHORTCUT: bool = os.getenv("ENABLE_PROBE_SHORTCUT", "true").lower() == "true"
# Closes the probe → archive loop with a tiny-experiment proxy + Bayesian credit.
# Adds ~1 LLM call per probe-passing idea. Default off so existing flows are unchanged.
ENABLE_EXECUTION_REVISION: bool = os.getenv("ENABLE_EXECUTION_REVISION", "false").lower() == "true"
EXECUTION_REVISION_SAMPLE_SIZE: int = int(os.getenv("EXECUTION_REVISION_SAMPLE_SIZE", "1000"))
EXECUTION_REVISION_N_SEEDS: int = int(os.getenv("EXECUTION_REVISION_N_SEEDS", "1"))

# Set by Pipeline.run() when the user opts into interactive control.
# base_agent's _observe_llm_call decorator reads this to know which
# RuntimeController to feed call results into. Lives at module scope so
# we don't have to plumb the controller through every agent constructor.
_current_runtime_controller: Optional[Any] = None
# Use aiprimetech.io proxy by default (same as ChatApp PhD project).
# Override with ANTHROPIC_BASE_URL=https://api.anthropic.com for direct Anthropic.
ANTHROPIC_BASE_URL: str = os.getenv("ANTHROPIC_BASE_URL", "https://aiprimetech.io")

SEMANTIC_SCHOLAR_API_KEY: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# ── Model selection (default per provider) ────────────────────────────────────
_DEFAULT_MODELS = {
    "deepseek": "deepseek-chat",
    "openai": "gpt-4o",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
    "azure": "DeepSeek-V3.2-Speciale",
    "anthropic": "claude-sonnet-4-6",
    # Default to moonshot-v1-32k (broadly compatible chat model). For
    # frontier reasoning, switch IDEAGRAPH_MODEL=kimi-k2.6 or kimi-k2.5;
    # base_agent auto-coerces temperature=1.0 for those.
    "kimi": "moonshot-v1-32k",
    # xAI Grok — DIFFERENT from `groq` above (Groq is fast Llama inference,
    # xAI is the api.x.ai endpoint). Endpoint is OpenAI-compatible at
    # https://api.x.ai/v1. Auth key starts with `xai-...`. NOTE: xAI
    # hosts both Grok AND DeepSeek V4 — your account/tier determines
    # which subset works. Default to deepseek-v4-pro because most
    # standard xai- keys hit that lineup; Grok models require a
    # higher account tier.
    "xai": "deepseek-v4-pro",
}
MODEL: str = os.getenv("IDEAGRAPH_MODEL", _DEFAULT_MODELS.get(PROVIDER, "deepseek-chat"))

# All supported providers (used by UI for dropdown)
SUPPORTED_PROVIDERS = list(_DEFAULT_MODELS.keys())

# Curated list of known Gemini-family models surfaced in the LLM Provider
# admin tab + sidebar dropdown when provider == 'gemini'. Newest first;
# the Anthropic-style dropdown lets users pick without having to know
# the exact API names.
#
# Categories:
#   - Text/multimodal LLMs (default chat use)
#   - Research-agent variants (Deep Research Pro Preview)
#   - Coding-agent variants (Antigravity)
#
# Model names are best-effort per Google's docs at
# https://ai.google.dev/gemini-api/docs/models — use the
# "📋 List models" diagnostic in the Visual Rendering tab if the API
# returns 404 for any of these.
# Curated list of known DeepSeek-family models surfaced in the LLM
# Provider admin tab when provider == 'deepseek'. The catalog gives
# users a one-click dropdown instead of having to memorize API model
# names (matches the Anthropic / Gemini UX). 'deepseek-chat' always
# points to the latest non-reasoning model; 'deepseek-reasoner' points
# to the latest R1-family reasoning model.
#
# Reference: https://api-docs.deepseek.com/quick_start/pricing
DEEPSEEK_KNOWN_MODELS: list = [
    # Production aliases — DeepSeek auto-routes to the newest version
    "deepseek-chat",       # latest non-reasoning (V3.x → V4 when GA)
    "deepseek-reasoner",   # latest reasoning (R1.x) — chain-of-thought
    # Next-gen / version-pinned. deepseek-v4 is the upcoming flagship —
    # if the API returns 404 for this name, `deepseek-chat` still routes
    # to whatever is the current production model.
    "deepseek-v4",
    "deepseek-v3.2",
    "deepseek-v3.1",
    "deepseek-v3",
    "deepseek-r1.1",
    "deepseek-r1",
    # Legacy / specialized (may 404 on newer accounts — kept for
    # reference so existing .env values resolve in the dropdown)
    "deepseek-coder",
]

GEMINI_KNOWN_MODELS: list = [
    # Gemini 3 (newest — paid AI Studio tier)
    "gemini-3.1-pro",
    "gemini-3-pro",
    "gemini-3-pro-preview",
    # Gemini 2.5
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    # Gemini 2.0
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-exp",
    # Specialized variants (paid tier — names best-effort)
    "gemini-deep-research-pro-preview",   # Deep Research Pro Preview
    "gemini-antigravity-preview",          # Antigravity coding agent
    # Gemini 1.5 (legacy but still widely available)
    "gemini-1.5-pro",
    "gemini-1.5-pro-002",
    "gemini-1.5-flash",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-8b",
]

# Per-provider cost rates (USD per million tokens) for budget estimation.
# Conservative upper bounds (cache-miss pricing).
COST_RATES = {
    "deepseek":  {"input": 0.27, "output": 1.10},
    "openai":    {"input": 2.50, "output": 10.00},
    "groq":      {"input": 0.59, "output": 0.79},
    "gemini":    {"input": 0.10, "output": 0.40},
    "azure":     {"input": 0.27, "output": 1.10},  # DeepSeek pricing on Azure
    "anthropic": {"input": 3.00, "output": 15.00},  # Sonnet default; opus higher, haiku lower
    "kimi":      {"input": 0.60, "output": 2.50},  # Moonshot kimi-k2 conservative upper bound
    "xai":       {"input": 2.00, "output": 10.00},  # xAI Grok grok-2 / grok-3 (similar to GPT-4o tier)
}

# ── DAG construction parameters ───────────────────────────────────────────────
SEED_QUERIES: int = int(os.getenv("SEED_QUERIES", "5"))
MAX_NODES: int = int(os.getenv("MAX_NODES", "50"))
DEPTH: int = int(os.getenv("DEPTH", "4"))
FORWARD_BRANCH: int = int(os.getenv("FORWARD_BRANCH", "3"))
BACKWARD_BRANCH: int = int(os.getenv("BACKWARD_BRANCH", "2"))
LATERAL_BRANCH: int = int(os.getenv("LATERAL_BRANCH", "1"))
SEEDS: int = int(os.getenv("SEEDS", "10"))

# ── Quality-Diversity parameters ──────────────────────────────────────────────
METHODS: int = 7          # number of methodology types (rows)
NOVELTY_LEVELS: int = 3   # number of novelty levels (columns)
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "20"))
COVERAGE_THRESHOLD: float = float(os.getenv("COVERAGE_THRESHOLD", "0.5"))
MAX_PARALLEL_CELLS: int = int(os.getenv("MAX_PARALLEL_CELLS", "6"))  # concurrent idea generation

# ── Debate Arena parameters ──────────────────────────────────────────────────
DEBATE_ENABLED: bool = os.getenv("DEBATE_ENABLED", "true").lower() == "true"
DEBATE_ROUNDS_PER_MATCH: int = int(os.getenv("DEBATE_ROUNDS", "2"))
DEBATE_MAX_IDEAS: int = int(os.getenv("DEBATE_MAX_IDEAS", "8"))
DEBATE_BUDGET_FRACTION: float = 0.3  # fraction of total budget for debate

# ── Paper Generator parameters ───────────────────────────────────────────────
PAPER_MAX_TOKENS: int = 4096
PAPER_SECTIONS = ["Abstract", "Introduction", "Related Work", "Method", "Expected Results", "Limitations"]

# ── Cross-Domain Synthesis parameters ────────────────────────────────────────
CROSS_DOMAIN_MAX_RUNS: int = int(os.getenv("CROSS_DOMAIN_MAX_RUNS", "3"))
CROSS_DOMAIN_SYNTHESIS_COUNT: int = 3

# ── Creative Optimization ────────────────────────────────────────────────────
ENABLE_CREATIVE_OPTIMIZATION: bool = os.getenv("ENABLE_CREATIVE_OPTIMIZATION", "true").lower() == "true"
ENABLE_PROMPT_EVOLUTION: bool = os.getenv("ENABLE_PROMPT_EVOLUTION", "true").lower() == "true"
ENABLE_THOMPSON_SAMPLING: bool = os.getenv("ENABLE_THOMPSON_SAMPLING", "true").lower() == "true"
ENABLE_ANNEALING: bool = os.getenv("ENABLE_ANNEALING", "true").lower() == "true"
ENABLE_PARETO_SELECTION: bool = os.getenv("ENABLE_PARETO_SELECTION", "true").lower() == "true"
ENABLE_CASCADE_ROUTING: bool = os.getenv("ENABLE_CASCADE_ROUTING", "false").lower() == "true"
ENABLE_CURIOSITY_EXPLORATION: bool = os.getenv("ENABLE_CURIOSITY_EXPLORATION", "true").lower() == "true"
ENABLE_MCTS_ROUTING: bool = os.getenv("ENABLE_MCTS_ROUTING", "true").lower() == "true"
ENABLE_ADVERSARIAL_TESTING: bool = os.getenv("ENABLE_ADVERSARIAL_TESTING", "true").lower() == "true"
ENABLE_ELO_RANKING: bool = os.getenv("ENABLE_ELO_RANKING", "true").lower() == "true"
ENABLE_PBT: bool = os.getenv("ENABLE_PBT", "true").lower() == "true"
ANNEALING_SCHEDULE: str = os.getenv("ANNEALING_SCHEDULE", "cosine")  # linear, exponential, cosine
ANNEALING_T_MAX: float = float(os.getenv("ANNEALING_T_MAX", "0.9"))
ANNEALING_T_MIN: float = float(os.getenv("ANNEALING_T_MIN", "0.2"))
PBT_POPULATION_SIZE: int = int(os.getenv("PBT_POPULATION_SIZE", "6"))
ELO_MAX_MATCHES: int = int(os.getenv("ELO_MAX_MATCHES", "6"))
ADVERSARIAL_MAX_ATTACKS: int = int(os.getenv("ADVERSARIAL_MAX_ATTACKS", "3"))
CURIOSITY_WEIGHT: float = float(os.getenv("CURIOSITY_WEIGHT", "0.3"))

# ── Deep Optimization ────────────────────────────────────────────────────────
ENABLE_DEEP_OPTIMIZATION: bool = os.getenv("ENABLE_DEEP_OPTIMIZATION", "true").lower() == "true"
ENABLE_ENSEMBLE_DISTILLATION: bool = os.getenv("ENABLE_ENSEMBLE_DISTILLATION", "false").lower() == "true"
ENABLE_MULTI_FIDELITY: bool = os.getenv("ENABLE_MULTI_FIDELITY", "true").lower() == "true"
ENABLE_BAYESIAN_TUNING: bool = os.getenv("ENABLE_BAYESIAN_TUNING", "true").lower() == "true"
ENABLE_KNAPSACK_BUDGET: bool = os.getenv("ENABLE_KNAPSACK_BUDGET", "true").lower() == "true"
ENABLE_PROGRESSIVE_ELABORATION: bool = os.getenv("ENABLE_PROGRESSIVE_ELABORATION", "false").lower() == "true"
ENABLE_CONTRASTIVE_PROMPTS: bool = os.getenv("ENABLE_CONTRASTIVE_PROMPTS", "true").lower() == "true"
ENABLE_ATTENTION_ROLLBACK: bool = os.getenv("ENABLE_ATTENTION_ROLLBACK", "true").lower() == "true"
ENABLE_RESPONSE_DISTILLATION: bool = os.getenv("ENABLE_RESPONSE_DISTILLATION", "true").lower() == "true"
ENABLE_ENTROPY_REGULARIZATION: bool = os.getenv("ENABLE_ENTROPY_REGULARIZATION", "true").lower() == "true"
ENABLE_REWARD_SHAPING: bool = os.getenv("ENABLE_REWARD_SHAPING", "true").lower() == "true"
ENABLE_TOKEN_PROJECTION: bool = os.getenv("ENABLE_TOKEN_PROJECTION", "true").lower() == "true"
ENABLE_DYNAMIC_BATCH_SIZE: bool = os.getenv("ENABLE_DYNAMIC_BATCH_SIZE", "true").lower() == "true"
ENSEMBLE_SIZE: int = int(os.getenv("ENSEMBLE_SIZE", "3"))
MULTI_FIDELITY_HALVING_RATE: float = float(os.getenv("MULTI_FIDELITY_HALVING_RATE", "3.0"))
ENTROPY_MIN_RATIO: float = float(os.getenv("ENTROPY_MIN_RATIO", "0.6"))
KNAPSACK_BUDGET_TOKENS_K: float = float(os.getenv("KNAPSACK_BUDGET_TOKENS_K", "30.0"))
EARLY_STOP_BUDGET_PCT: float = float(os.getenv("EARLY_STOP_BUDGET_PCT", "10.0"))

# ── Infrastructure Optimization ──────────────────────────────────────────────
ENABLE_INFRA_OPTIMIZATION: bool = os.getenv("ENABLE_INFRA_OPTIMIZATION", "true").lower() == "true"
# Disk cache persists LLM responses across process restarts, giving ~30-60%
# cost reduction on dev loops where the same prompts recur. Survives restarts
# and is NOT cleared by reset_caches() (see agents/base_agent.py). Set to
# "false" in .env if you need strictly fresh responses every run.
ENABLE_DISK_CACHE: bool = os.getenv("ENABLE_DISK_CACHE", "true").lower() == "true"
# Semantic cache matches near-duplicate prompts via n-gram Jaccard similarity.
# Safer than exact-match for prompts with small wording variations.
ENABLE_SEMANTIC_CACHE: bool = os.getenv("ENABLE_SEMANTIC_CACHE", "true").lower() == "true"
ENABLE_PROVIDER_ROUTER: bool = os.getenv("ENABLE_PROVIDER_ROUTER", "true").lower() == "true"
ENABLE_ERROR_CLASSIFIER: bool = os.getenv("ENABLE_ERROR_CLASSIFIER", "true").lower() == "true"
ENABLE_COST_ATTRIBUTION: bool = os.getenv("ENABLE_COST_ATTRIBUTION", "true").lower() == "true"
ENABLE_STRUCTURED_LOGGING: bool = os.getenv("ENABLE_STRUCTURED_LOGGING", "true").lower() == "true"
ENABLE_DRY_RUN: bool = os.getenv("ENABLE_DRY_RUN", "false").lower() == "true"
ENABLE_ABLATION: bool = os.getenv("ENABLE_ABLATION", "false").lower() == "true"
ENABLE_RESOURCE_MONITOR: bool = os.getenv("ENABLE_RESOURCE_MONITOR", "true").lower() == "true"
ENABLE_DAG_EXECUTOR: bool = os.getenv("ENABLE_DAG_EXECUTOR", "false").lower() == "true"
ENABLE_ARTIFACT_STORE: bool = os.getenv("ENABLE_ARTIFACT_STORE", "true").lower() == "true"
DISK_CACHE_MAX_MB: float = float(os.getenv("DISK_CACHE_MAX_MB", "50"))
SEMANTIC_CACHE_THRESHOLD: float = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.75"))
MEMORY_LIMIT_MB: float = float(os.getenv("MEMORY_LIMIT_MB", "2048"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ── Meta Optimization ────────────────────────────────────────────────────────
ENABLE_META_OPTIMIZATION: bool = os.getenv("ENABLE_META_OPTIMIZATION", "true").lower() == "true"
ENABLE_KALMAN_FILTER: bool = os.getenv("ENABLE_KALMAN_FILTER", "true").lower() == "true"
ENABLE_MUTUAL_INFO: bool = os.getenv("ENABLE_MUTUAL_INFO", "true").lower() == "true"
ENABLE_NASH_CONSENSUS: bool = os.getenv("ENABLE_NASH_CONSENSUS", "true").lower() == "true"
ENABLE_WORKING_MEMORY: bool = os.getenv("ENABLE_WORKING_MEMORY", "true").lower() == "true"
ENABLE_DEAD_STAGE_ELIMINATION: bool = os.getenv("ENABLE_DEAD_STAGE_ELIMINATION", "true").lower() == "true"
ENABLE_PREDICTIVE_GATE: bool = os.getenv("ENABLE_PREDICTIVE_GATE", "true").lower() == "true"
ENABLE_BLOOM_DEDUP: bool = os.getenv("ENABLE_BLOOM_DEDUP", "true").lower() == "true"
ENABLE_FORECASTING: bool = os.getenv("ENABLE_FORECASTING", "true").lower() == "true"
ENABLE_AB_TESTING: bool = os.getenv("ENABLE_AB_TESTING", "false").lower() == "true"
ENABLE_CAUSAL_ANALYSIS: bool = os.getenv("ENABLE_CAUSAL_ANALYSIS", "false").lower() == "true"
ENABLE_RESERVOIR_SAMPLING: bool = os.getenv("ENABLE_RESERVOIR_SAMPLING", "true").lower() == "true"
ENABLE_FEEDBACK_DETECTION: bool = os.getenv("ENABLE_FEEDBACK_DETECTION", "true").lower() == "true"
KALMAN_PROCESS_NOISE: float = float(os.getenv("KALMAN_PROCESS_NOISE", "0.01"))
KALMAN_MEASUREMENT_NOISE: float = float(os.getenv("KALMAN_MEASUREMENT_NOISE", "0.1"))
PREDICTIVE_GATE_THRESHOLD: float = float(os.getenv("PREDICTIVE_GATE_THRESHOLD", "0.3"))
BLOOM_EXPECTED_ITEMS: int = int(os.getenv("BLOOM_EXPECTED_ITEMS", "1000"))
RESERVOIR_SIZE: int = int(os.getenv("RESERVOIR_SIZE", "50"))
FEEDBACK_WINDOW: int = int(os.getenv("FEEDBACK_WINDOW", "4"))

# ── Quantum/Frontier Optimization ────────────────────────────────────────────
ENABLE_QUANTUM_OPTIMIZATION: bool = os.getenv("ENABLE_QUANTUM_OPTIMIZATION", "true").lower() == "true"
ENABLE_QUANTUM_ANNEALING: bool = os.getenv("ENABLE_QUANTUM_ANNEALING", "true").lower() == "true"
ENABLE_SWARM_OPTIMIZATION: bool = os.getenv("ENABLE_SWARM_OPTIMIZATION", "false").lower() == "true"
ENABLE_TOPOLOGICAL_DIVERSITY: bool = os.getenv("ENABLE_TOPOLOGICAL_DIVERSITY", "true").lower() == "true"
ENABLE_FRACTAL_BUDGET: bool = os.getenv("ENABLE_FRACTAL_BUDGET", "true").lower() == "true"
ENABLE_COGNITIVE_LOAD: bool = os.getenv("ENABLE_COGNITIVE_LOAD", "true").lower() == "true"
ENABLE_HYPERBAND: bool = os.getenv("ENABLE_HYPERBAND", "false").lower() == "true"
ENABLE_WASSERSTEIN: bool = os.getenv("ENABLE_WASSERSTEIN", "false").lower() == "true"
ENABLE_BIFURCATION: bool = os.getenv("ENABLE_BIFURCATION", "false").lower() == "true"
ENABLE_CONTEXTUAL_BANDIT: bool = os.getenv("ENABLE_CONTEXTUAL_BANDIT", "true").lower() == "true"
ENABLE_CMA_ES: bool = os.getenv("ENABLE_CMA_ES", "true").lower() == "true"
ENABLE_INFO_BOTTLENECK: bool = os.getenv("ENABLE_INFO_BOTTLENECK", "true").lower() == "true"
ENABLE_META_LEARNING: bool = os.getenv("ENABLE_META_LEARNING", "true").lower() == "true"
QUANTUM_ANNEAL_STEPS: int = int(os.getenv("QUANTUM_ANNEAL_STEPS", "200"))
SWARM_PARTICLES: int = int(os.getenv("SWARM_PARTICLES", "10"))
CMA_ES_POP_SIZE: int = int(os.getenv("CMA_ES_POP_SIZE", "8"))
CONTEXTUAL_BANDIT_ALPHA: float = float(os.getenv("CONTEXTUAL_BANDIT_ALPHA", "0.5"))

# ── Nature-Inspired Optimization ─────────────────────────────────────────────
ENABLE_NATURE_OPTIMIZATION: bool = os.getenv("ENABLE_NATURE_OPTIMIZATION", "true").lower() == "true"
ENABLE_ANT_COLONY: bool = os.getenv("ENABLE_ANT_COLONY", "true").lower() == "true"
ENABLE_IMMUNE_SYSTEM: bool = os.getenv("ENABLE_IMMUNE_SYSTEM", "true").lower() == "true"
ENABLE_GRAVITATIONAL: bool = os.getenv("ENABLE_GRAVITATIONAL", "false").lower() == "true"
ENABLE_DIFFUSION: bool = os.getenv("ENABLE_DIFFUSION", "false").lower() == "true"
ENABLE_VICKREY_AUCTION: bool = os.getenv("ENABLE_VICKREY_AUCTION", "true").lower() == "true"
ENABLE_WISDOM_OF_CROWDS: bool = os.getenv("ENABLE_WISDOM_OF_CROWDS", "true").lower() == "true"
ENABLE_MOMENTUM: bool = os.getenv("ENABLE_MOMENTUM", "true").lower() == "true"
ENABLE_SECRETARY: bool = os.getenv("ENABLE_SECRETARY", "false").lower() == "true"
ENABLE_COEVOLUTION: bool = os.getenv("ENABLE_COEVOLUTION", "true").lower() == "true"
ENABLE_THERMODYNAMICS: bool = os.getenv("ENABLE_THERMODYNAMICS", "true").lower() == "true"
ENABLE_SOCIAL_INFLUENCE: bool = os.getenv("ENABLE_SOCIAL_INFLUENCE", "true").lower() == "true"
ENABLE_CHAOTIC_EXPLORER: bool = os.getenv("ENABLE_CHAOTIC_EXPLORER", "true").lower() == "true"
ACO_EVAPORATION: float = float(os.getenv("ACO_EVAPORATION", "0.1"))
ACO_N_ANTS: int = int(os.getenv("ACO_N_ANTS", "10"))
IMMUNE_POP_SIZE: int = int(os.getenv("IMMUNE_POP_SIZE", "20"))
COEVOLUTION_CRITICS: int = int(os.getenv("COEVOLUTION_CRITICS", "3"))
MOMENTUM_BETA1: float = float(os.getenv("MOMENTUM_BETA1", "0.9"))
CHAOS_R: float = float(os.getenv("CHAOS_R", "3.99"))

# ── Cognitive Optimization ───────────────────────────────────────────────────
ENABLE_COGNITIVE_OPTIMIZATION: bool = os.getenv("ENABLE_COGNITIVE_OPTIMIZATION", "true").lower() == "true"
ENABLE_EPISODIC_MEMORY: bool = os.getenv("ENABLE_EPISODIC_MEMORY", "true").lower() == "true"
ENABLE_SEMANTIC_MEMORY: bool = os.getenv("ENABLE_SEMANTIC_MEMORY", "true").lower() == "true"
ENABLE_SHAPLEY_VALUES: bool = os.getenv("ENABLE_SHAPLEY_VALUES", "true").lower() == "true"
ENABLE_COT_OPTIMIZATION: bool = os.getenv("ENABLE_COT_OPTIMIZATION", "true").lower() == "true"
ENABLE_MANIFOLD_EXPLORER: bool = os.getenv("ENABLE_MANIFOLD_EXPLORER", "false").lower() == "true"
ENABLE_CHAOS_ENGINEERING: bool = os.getenv("ENABLE_CHAOS_ENGINEERING", "false").lower() == "true"
ENABLE_CANARY_DEPLOY: bool = os.getenv("ENABLE_CANARY_DEPLOY", "false").lower() == "true"
ENABLE_PRIORITY_AGING: bool = os.getenv("ENABLE_PRIORITY_AGING", "true").lower() == "true"
ENABLE_TIME_BOXING: bool = os.getenv("ENABLE_TIME_BOXING", "true").lower() == "true"
ENABLE_LINGUISTIC_ANALYSIS: bool = os.getenv("ENABLE_LINGUISTIC_ANALYSIS", "true").lower() == "true"
ENABLE_PROMPT_ALGEBRA: bool = os.getenv("ENABLE_PROMPT_ALGEBRA", "true").lower() == "true"
ENABLE_GEODETIC_DISTANCE: bool = os.getenv("ENABLE_GEODETIC_DISTANCE", "false").lower() == "true"
SHAPLEY_SAMPLES: int = int(os.getenv("SHAPLEY_SAMPLES", "100"))
TIME_BOX_TOTAL_S: float = float(os.getenv("TIME_BOX_TOTAL_S", "600"))
CHAOS_FAULT_PROB: float = float(os.getenv("CHAOS_FAULT_PROB", "0.05"))
EPISODIC_MAX_EPISODES: int = int(os.getenv("EPISODIC_MAX_EPISODES", "100"))
GEODETIC_K_NEIGHBORS: int = int(os.getenv("GEODETIC_K_NEIGHBORS", "5"))

# ── Aesthetic/Cross-Domain Optimization ──────────────────────────────────────
ENABLE_AESTHETIC_OPTIMIZATION: bool = os.getenv("ENABLE_AESTHETIC_OPTIMIZATION", "true").lower() == "true"
ENABLE_PORTFOLIO: bool = os.getenv("ENABLE_PORTFOLIO", "true").lower() == "true"
ENABLE_SPACED_REPETITION: bool = os.getenv("ENABLE_SPACED_REPETITION", "true").lower() == "true"
ENABLE_DIALECTICAL: bool = os.getenv("ENABLE_DIALECTICAL", "true").lower() == "true"
ENABLE_NARRATIVE_ARC: bool = os.getenv("ENABLE_NARRATIVE_ARC", "true").lower() == "true"
ENABLE_RISK_PARITY: bool = os.getenv("ENABLE_RISK_PARITY", "true").lower() == "true"
ENABLE_ZPD: bool = os.getenv("ENABLE_ZPD", "true").lower() == "true"
ENABLE_DESIRE_LINES: bool = os.getenv("ENABLE_DESIRE_LINES", "true").lower() == "true"
ENABLE_HARMONIC: bool = os.getenv("ENABLE_HARMONIC", "true").lower() == "true"
ENABLE_SABERMETRIC: bool = os.getenv("ENABLE_SABERMETRIC", "true").lower() == "true"
ENABLE_LOTKA_VOLTERRA: bool = os.getenv("ENABLE_LOTKA_VOLTERRA", "true").lower() == "true"
ENABLE_GOLDEN_RATIO: bool = os.getenv("ENABLE_GOLDEN_RATIO", "true").lower() == "true"
ENABLE_HEROS_JOURNEY: bool = os.getenv("ENABLE_HEROS_JOURNEY", "true").lower() == "true"
PORTFOLIO_RISK_FREE_RATE: float = float(os.getenv("PORTFOLIO_RISK_FREE_RATE", "0.1"))
ZPD_LOW: float = float(os.getenv("ZPD_LOW", "0.6"))
ZPD_HIGH: float = float(os.getenv("ZPD_HIGH", "0.8"))
LOTKA_ALPHA: float = float(os.getenv("LOTKA_ALPHA", "0.5"))

# ── Agent Swarm Optimization (Layer 10) ──────────────────────────────────────
ENABLE_SWARM_OPTIMIZATION_LAYER: bool = os.getenv("ENABLE_SWARM_OPTIMIZATION_LAYER", "true").lower() == "true"
ENABLE_AGENT_POOL: bool = os.getenv("ENABLE_AGENT_POOL", "true").lower() == "true"
ENABLE_MESSAGE_BUS: bool = os.getenv("ENABLE_MESSAGE_BUS", "true").lower() == "true"
ENABLE_SHARED_BLACKBOARD: bool = os.getenv("ENABLE_SHARED_BLACKBOARD", "true").lower() == "true"
ENABLE_SWARM_CONSENSUS: bool = os.getenv("ENABLE_SWARM_CONSENSUS", "true").lower() == "true"
ENABLE_HIERARCHICAL_COORDINATOR: bool = os.getenv("ENABLE_HIERARCHICAL_COORDINATOR", "true").lower() == "true"
ENABLE_SPECIALIST_ROUTER: bool = os.getenv("ENABLE_SPECIALIST_ROUTER", "true").lower() == "true"
ENABLE_DYNAMIC_TEAM_FORMER: bool = os.getenv("ENABLE_DYNAMIC_TEAM_FORMER", "true").lower() == "true"
ENABLE_STIGMERGY: bool = os.getenv("ENABLE_STIGMERGY", "true").lower() == "true"
ENABLE_AGENT_NEGOTIATOR: bool = os.getenv("ENABLE_AGENT_NEGOTIATOR", "true").lower() == "true"
ENABLE_SWARM_MEMORY_POOL: bool = os.getenv("ENABLE_SWARM_MEMORY_POOL", "true").lower() == "true"
ENABLE_EMERGENCE_DETECTOR: bool = os.getenv("ENABLE_EMERGENCE_DETECTOR", "true").lower() == "true"
ENABLE_MULTI_AGENT_DEBATE: bool = os.getenv("ENABLE_MULTI_AGENT_DEBATE", "true").lower() == "true"
SWARM_MAX_AGENTS: int = int(os.getenv("SWARM_MAX_AGENTS", "12"))
SWARM_AGENT_IDLE_TIMEOUT_S: float = float(os.getenv("SWARM_AGENT_IDLE_TIMEOUT_S", "120"))
SWARM_HIERARCHY_MAX_DEPTH: int = int(os.getenv("SWARM_HIERARCHY_MAX_DEPTH", "3"))
SWARM_CONSENSUS_DEFAULT: str = os.getenv("SWARM_CONSENSUS_DEFAULT", "weighted_average")
SWARM_STIGMERGY_EVAPORATION: float = float(os.getenv("SWARM_STIGMERGY_EVAPORATION", "0.05"))

# ── Systems/Reliability Optimization (Layer 11) ─────────────────────────────
ENABLE_SYSTEMS_OPTIMIZATION: bool = os.getenv("ENABLE_SYSTEMS_OPTIMIZATION", "true").lower() == "true"
ENABLE_RAFT_CONSENSUS: bool = os.getenv("ENABLE_RAFT_CONSENSUS", "true").lower() == "true"
ENABLE_BACKPRESSURE: bool = os.getenv("ENABLE_BACKPRESSURE", "true").lower() == "true"
ENABLE_BULKHEAD: bool = os.getenv("ENABLE_BULKHEAD", "true").lower() == "true"
ENABLE_LOAD_SHEDDING: bool = os.getenv("ENABLE_LOAD_SHEDDING", "true").lower() == "true"
ENABLE_RATE_LIMITER: bool = os.getenv("ENABLE_RATE_LIMITER", "true").lower() == "true"
ENABLE_KANBAN: bool = os.getenv("ENABLE_KANBAN", "true").lower() == "true"
ENABLE_CRITICAL_PATH: bool = os.getenv("ENABLE_CRITICAL_PATH", "true").lower() == "true"
ENABLE_LOOP_HOISTING: bool = os.getenv("ENABLE_LOOP_HOISTING", "true").lower() == "true"
ENABLE_CENTRALITY: bool = os.getenv("ENABLE_CENTRALITY", "true").lower() == "true"
ENABLE_PID_CONTROLLER: bool = os.getenv("ENABLE_PID_CONTROLLER", "true").lower() == "true"
ENABLE_BACKOFF_POOL: bool = os.getenv("ENABLE_BACKOFF_POOL", "true").lower() == "true"
ENABLE_CHECKPOINT_RECOVERY: bool = os.getenv("ENABLE_CHECKPOINT_RECOVERY", "true").lower() == "true"
PID_KP: float = float(os.getenv("PID_KP", "0.5"))
PID_KI: float = float(os.getenv("PID_KI", "0.1"))
PID_KD: float = float(os.getenv("PID_KD", "0.05"))
PID_TARGET_QUALITY: float = float(os.getenv("PID_TARGET_QUALITY", "0.7"))
KANBAN_DEFAULT_WIP: int = int(os.getenv("KANBAN_DEFAULT_WIP", "3"))
BACKPRESSURE_MAX_QUEUE: int = int(os.getenv("BACKPRESSURE_MAX_QUEUE", "10"))

# ── Advanced Optimization ────────────────────────────────────────────────────
ENABLE_PROMPT_COMPRESSION: bool = os.getenv("ENABLE_PROMPT_COMPRESSION", "true").lower() == "true"
ENABLE_CIRCUIT_BREAKER: bool = os.getenv("ENABLE_CIRCUIT_BREAKER", "true").lower() == "true"
ENABLE_SPECULATIVE_EXEC: bool = os.getenv("ENABLE_SPECULATIVE_EXEC", "true").lower() == "true"
ENABLE_WARM_CACHE: bool = os.getenv("ENABLE_WARM_CACHE", "true").lower() == "true"
MAX_CONCURRENT_WORKERS: int = int(os.getenv("MAX_CONCURRENT_WORKERS", "6"))
CIRCUIT_BREAKER_THRESHOLD: int = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))
CIRCUIT_BREAKER_RECOVERY_S: float = float(os.getenv("CIRCUIT_BREAKER_RECOVERY_S", "60"))
WARM_CACHE_MAX_ENTRIES: int = int(os.getenv("WARM_CACHE_MAX_ENTRIES", "64"))
WARM_CACHE_TTL_S: float = float(os.getenv("WARM_CACHE_TTL_S", "1800"))

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", str(Path(__file__).parent / "output")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
