# SECURITY

## URGENT: Rotate exposed API keys

The values in `.env` have been visible in your local working copy and any
backups/prior-tool contexts. Before putting this repo on GitHub or any
shared surface, rotate:

### 1. DeepSeek
1. Go to https://platform.deepseek.com/api_keys
2. Revoke the key starting with `sk-fb2bd69b...`
3. Generate a new key
4. Update `DEEPSEEK_API_KEY` in your local `.env`

### 2. Azure AI
1. Open the Azure portal → your AI service resource `ai-nlp26-2003-resource`
2. "Keys and Endpoint" → click **"Regenerate Key 1"**
3. Update `AZURE_API_KEY` in your local `.env`

### 3. Before any `git init` / push
- `.gitignore` already excludes `.env` (do **not** remove that rule)
- Verify: `git check-ignore -v .env` should print the ignore rule
- If you've already pushed `.env` anywhere, use
  [git-filter-repo](https://github.com/newren/git-filter-repo)
  to purge it from history — a force-push alone is not enough

## Production secret management

Once you're past local development:

- **Never** ship `.env` in a Docker image — `.dockerignore` already excludes it
- Inject secrets at runtime via:
  - Docker Swarm: `docker secret create ...`
  - Kubernetes: `Secret` resources + `envFrom: secretRef`
  - AWS/GCP/Azure: Secrets Manager / Secret Manager / Key Vault
- Rotate all secrets every 90 days minimum
- Scope keys per environment (dev keys ≠ staging keys ≠ prod keys)

## Security posture implemented in this codebase

| Control | Where | Status |
|---|---|---|
| Password strength (≥12 chars, letter+digit, blacklist) | `production_optimization.validate_password` + `auth_ui.py` | ✓ |
| Cryptographic session tokens (256-bit, 7-day TTL, SHA-256 hash check) | `auth_ui._save_remember_token` | ✓ |
| Login rate-limit (per-username) | `auth_ui.show_auth_page` | ✓ |
| Prompt-injection detection on topic input | `production_optimization._INJECTION_PATTERNS` | ✓ |
| Per-tier budget + iteration hard caps | `production_optimization.validate_run_input` | ✓ |
| API rate limiting (sliding window + token bucket) | `production_optimization.RateLimiter` | ✓ |
| CORS allowlist (env-configurable, no `*`) | `api.py::_ALLOWED_ORIGINS` | ✓ |
| Optional API bearer token (constant-time compare) | `api.py::_require_auth` | ✓ |
| Atomic subscription-quota enforcement | `production_optimization.QuotaEnforcer` | ✓ |
| Circuit breaker per LLM provider | `production_optimization.CircuitBreaker` | ✓ |
| Global + per-user concurrency caps | `production_optimization.ConcurrencyGuard` | ✓ |
| Per-user / per-run cost attribution + hard cap | `production_optimization.CostTracker` | ✓ |
| Structured JSON logs with secret redaction | `observability.StructuredLogger` | ✓ |
| Prometheus metrics endpoint (`/metrics`) | `api.py` | ✓ |
| Bounded task queue (replaces unbounded threads) | `task_queue.ThreadQueueBackend` | ✓ |
| Docker non-root runtime user | `Dockerfile` (uid 1000) | ✓ |
| `.env` excluded from git + Docker builds | `.gitignore` + `.dockerignore` | ✓ |
| Pytest suite (74 tests, passes in <5s) | `tests/` | ✓ |

## Still open (requires infrastructure work)

- Migrate SQLite → PostgreSQL (compose file is ready; `db.py` still uses SQLite)
- Swap threading backend → Celery in production (env: `TASK_QUEUE_BACKEND=celery`)
- Replace Streamlit frontend with React/Next.js for multi-server horizontal scale
- 2FA / TOTP for paid accounts
- SSO (Google / GitHub / SAML) — required for enterprise sales
- SOC 2 + GDPR audit
- External penetration test before public launch

## Reporting vulnerabilities

If you discover a vulnerability, email security@ideagraph.example.com (replace
with real address before launch). Do not file public GitHub issues for security
problems.
