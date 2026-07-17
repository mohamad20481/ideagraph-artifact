# syntax=docker/dockerfile:1.6
# ──────────────────────────────────────────────────────────────────────────
# IdeaGraph — multi-stage, slim, non-root production image.
# Build:   docker build -t ideagraph:latest .
# Run:     docker run --env-file .env -p 8510:8510 ideagraph:latest
# ──────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder — install wheels into a throwaway layer ─────────────
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build deps only; not copied into the final image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --prefix=/install -r requirements.txt \
 && pip install --prefix=/install \
        "celery[redis]>=5.3" \
        "redis>=5.0" \
        "psycopg2-binary>=2.9" \
        "gunicorn>=22.0"


# ── Stage 2: runtime — minimal image, non-root, app only ─────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/install/bin:$PATH \
    PYTHONPATH=/install/lib/python3.12/site-packages \
    STREAMLIT_SERVER_PORT=8510 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Runtime system deps (curl for healthcheck, tini for proper signal handling).
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl tini \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd -g 1000 app \
 && useradd -u 1000 -g 1000 -m -s /bin/bash app

COPY --from=builder /install /install

# App code (owned by the non-root user).
WORKDIR /app
COPY --chown=app:app . /app
RUN mkdir -p /app/data /app/output \
 && chown -R app:app /app/data /app/output

USER app

EXPOSE 8510 8502

# Healthcheck hits the Streamlit internal endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${STREAMLIT_SERVER_PORT}/_stcore/health" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
# Default command: Streamlit UI. Override with `api` or `worker` (see entrypoint).
CMD ["bash", "-lc", "streamlit run app.py --server.port=${STREAMLIT_SERVER_PORT} --server.address=0.0.0.0"]
