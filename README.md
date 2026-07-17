# IdeaGraph

**Automated research-ideation pipeline.** Generates research ideas with Quality-Diversity (MAP-Elites) coverage of a 7×3 methodology×novelty archive, then refines them through chat-to-optimize, adversarial debate, and visual paper-style abstracts.

Built around a pluggable LLM provider layer (DeepSeek, OpenAI, Groq, Gemini, Azure, Anthropic, Kimi/Moonshot, xAI Grok) with an OpenAI-compatible dispatcher, Streamlit UI, and FastAPI gateway.

---

## Highlights

- **🧠 7×3 QD archive** — Quality-Diversity ideation across (methodology × novelty) cells; coverage typically 40–60% on a $0.05–$0.10 run
- **🔌 8 LLM providers** — DeepSeek, OpenAI, Groq, Gemini, Azure, Anthropic, Kimi (Moonshot), xAI Grok — switchable at runtime via admin dashboard
- **💬 Chat-with-result** — per-result conversational refinement, persistent across sessions, with click-to-jump idea index
- **🎨 Visual abstracts** — paper-figure-style images via FLUX, Google Imagen, Gemini Flash Image, or Veo 3 video clips
- **🎛️ Interactive QD (iQD)** — freeze / prune cells, directed crossover between any two parent ideas under a user constraint
- **🔮 Bayesian Surprise** — information-theoretic novelty (1 − cos(P, P|H)) as a sort mode, distinct from LLM-judge novelty bias
- **⚔️ Debate Fitness** — σ(τ · margin) from T rounds of Proposer vs Critic dialogue per idea
- **💳 Billing** — Stripe Checkout + Customer Portal + admin tier-override panel (4 tiers: free / pro / team / enterprise)
- **🚦 Concurrency safety** — run-scoped release tokens, admin Active Runs panel for stuck-slot recovery
- **🧪 270+ tests** — unit + integration + adversarial-verification workflows

## Quick start

```bash
# 1. Clone + install
git clone https://github.com/YOUR-USERNAME/ideagraph.git
cd ideagraph
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env: set at minimum IDEAGRAPH_PROVIDER and the matching API key.

# 3. Run the Streamlit app
streamlit run app.py
# → http://localhost:8510

# OR run a one-shot pipeline from CLI:
python main.py --topic "few-shot learning for medical NLP" --budget 0.10 --iterations 3
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        IdeaGraph pipeline                           │
└────────────────────────────────────┬────────────────────────────────┘
                                     ▼
                      ┌──────────────────────────┐
                      │   Multi-source corpus    │◄── arXiv / S2 / DOI
                      └──────────────┬───────────┘
                                     ▼
                      ┌──────────────────────────┐
                      │     DAG knowledge graph  │
                      └──────────────┬───────────┘
                                     ▼
                      ┌──────────────────────────┐
                      │    Ideation agents (LLM) │── ←─┐
                      └──────────────┬───────────┘     │
                                     ▼                 │
                      ┌──────────────────────────┐     │ refine loop
                      │ 7×3 MAP-Elites archive   │ ────┘
                      └──────────────┬───────────┘
                                     ▼
                      ┌──────────────────────────┐
                      │  Probe scoring + sort    │
                      └──────────────┬───────────┘
                                     ▼
                      ┌──────────────────────────┐
                      │       Streamlit UI       │
                      └──────────────────────────┘
```

### Provider dispatcher

`claude_provider.py` exposes `get_claude_client()` which returns:
- `ClaudeClient` (Anthropic native `/v1/messages` with prompt caching) for `provider=anthropic`
- `OpenAICompatClient` pointed at each provider's `base_url` + key for the rest

The dispatcher table at [`claude_provider.py:480`](claude_provider.py) maps each provider to its `(api_key, base_url, default_model)`. Add a new provider by extending that table plus the catalog in [`config.py`](config.py).

## Configuration

See [`.env.example`](.env.example) for the complete list. Minimum viable config:

```bash
IDEAGRAPH_PROVIDER=deepseek          # or any of: openai groq gemini azure anthropic kimi xai
IDEAGRAPH_MODEL=deepseek-chat
DEEPSEEK_API_KEY=sk-...              # provider-specific key
```

## Subscription tiers (optional)

If you want to ship this as a SaaS, set up Stripe (test mode first):

1. Create 3 recurring products at [dashboard.stripe.com/test/products](https://dashboard.stripe.com/test/products): Pro $15/mo, Team $49/mo, Enterprise $299/mo
2. Paste the resulting `price_…` IDs into `.env` (`STRIPE_PRICE_PRO`, etc.)
3. Set `STRIPE_SECRET_KEY=sk_test_...` and `APP_BASE_URL=http://localhost:8510`
4. Restart Streamlit → **Admin → 💳 Billing** confirms connection

The return-flow polling model means no public webhook URL is required for localhost development.

## Tests

```bash
pytest tests/                     # full suite (~3-5 minutes)
pytest tests/test_release_callback.py -q   # one focused file
pytest -k "billing" -q            # by keyword
```

Test counts (as of the most recent build):
- ~270 tests covering: payments, account management, chat persistence, idea sorting,
  Bayesian Surprise, debate fitness, interactive iQD, concurrency reset, release-callback
  drift detection, provider dispatch, Grok image provider, billing, Stripe payments

All LLM and HTTP calls are mocked — the suite never hits real APIs.

## Documentation

- [`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md) — how to integrate IdeaGraph as a library
- [`SECURITY.md`](SECURITY.md) — reporting vulnerabilities, threat model
- [`PDF_GUIDE_GENERATION_PROMPT.md`](PDF_GUIDE_GENERATION_PROMPT.md) — the prompt used to generate the user guide

## License

Add your license file here.

## Acknowledgements

Built on the shoulders of: MAP-Elites (Cully et al.), Quality-Diversity literature, the OpenAI Python SDK, Streamlit, FastAPI, NetworkX, and the wider open LLM ecosystem.
