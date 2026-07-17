"""
api.py - REST API for programmatic IdeaGraph pipeline access.

Endpoints:
  POST /api/run          — Start a pipeline run (returns run_id)
  GET  /api/status/{id}  — Check run status + live progress
  GET  /api/results/{id} — Get full results
  GET  /api/ideas/{id}   — Get just the ideas
  GET  /api/runs         — List all runs
  GET  /api/health       — Health check

Usage:
  uvicorn api:app --port 8502
  # or
  python api.py

Example:
  curl -X POST http://localhost:8502/api/run \
    -H "Content-Type: application/json" \
    -d '{"topic": "transformer attention mechanisms", "budget_usd": 1.0}'

  curl http://localhost:8502/api/status/<run_id>
  curl http://localhost:8502/api/results/<run_id>
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Request, Header
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from production_optimization import (
    InputValidationError,
    get_circuit_breaker,
    get_concurrency_guard,
    get_cost_tracker,
    get_rate_limiter,
    health_snapshot,
    validate_run_input,
)

# CORS allowlist — comma-separated origins via env, defaults to localhost only.
_ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv(
        "API_ALLOWED_ORIGINS",
        "http://localhost:8510,http://localhost:3000",
    ).split(",") if o.strip()
]
# Optional bearer token for API auth (opt-in; set API_BEARER_TOKEN to enforce).
_API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "").strip()

# Run-retention: drop completed runs older than N seconds to bound memory.
_RUN_RETENTION_S = int(os.getenv("API_RUN_RETENTION_S", "3600"))
_RUN_MAX_ENTRIES = int(os.getenv("API_RUN_MAX_ENTRIES", "10000"))

# Storage for active/completed runs
_RUNS: Dict[str, Dict[str, Any]] = {}
_RUNS_LOCK = threading.Lock()


def _gc_runs() -> None:
    """Evict old completed runs so _RUNS doesn't grow unbounded."""
    now = time.time()
    with _RUNS_LOCK:
        if len(_RUNS) <= _RUN_MAX_ENTRIES and all(
            now - r.get("created_at", now) < _RUN_RETENTION_S
            for r in _RUNS.values()
        ):
            return
        # Prefer evicting completed/errored runs first.
        victims = [
            (rid, r) for rid, r in _RUNS.items()
            if r.get("status") in ("completed", "error")
            and now - r.get("completed_at", r.get("created_at", now)) > _RUN_RETENTION_S
        ]
        for rid, _ in victims:
            _RUNS.pop(rid, None)
        # If still oversize, drop the oldest completed/errored.
        if len(_RUNS) > _RUN_MAX_ENTRIES:
            completed = sorted(
                ((rid, r) for rid, r in _RUNS.items()
                 if r.get("status") in ("completed", "error")),
                key=lambda kv: kv[1].get("completed_at") or kv[1].get("created_at", 0),
            )
            for rid, _ in completed[: len(_RUNS) - _RUN_MAX_ENTRIES]:
                _RUNS.pop(rid, None)


if HAS_FASTAPI:

    class RunRequest(BaseModel):
        topic: str = Field(..., description="Research topic to explore")
        budget_usd: float = Field(2.0, description="API budget in USD")
        max_iterations: int = Field(10, description="Max ideation iterations")
        max_scientist_iterations: int = Field(2, description="Max scientist loop iterations")
        debate_enabled: bool = Field(False, description="Enable debate arena")
        execution_timeout: int = Field(600, description="Execution timeout (seconds)")
        provider: str = Field("", description="LLM provider override")
        model: str = Field("", description="Model override")

    class RunResponse(BaseModel):
        run_id: str
        status: str
        message: str

    class StatusResponse(BaseModel):
        run_id: str
        status: str
        progress: List[str] = []
        elapsed_s: float = 0
        error: Optional[str] = None

    app = FastAPI(
        title="IdeaGraph API",
        description="REST API for automated research ideation and experimentation",
        version="2.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    def _client_ip(request: Request) -> str:
        # Honour X-Forwarded-For when behind a trusted proxy/load-balancer.
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _require_auth(authorization: Optional[str]) -> None:
        """If API_BEARER_TOKEN is configured, require it. Otherwise allow (dev mode)."""
        if not _API_BEARER_TOKEN:
            return
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "Missing bearer token")
        token = authorization.split(" ", 1)[1].strip()
        # Constant-time comparison to thwart timing attacks.
        import hmac
        if not hmac.compare_digest(token, _API_BEARER_TOKEN):
            raise HTTPException(401, "Invalid bearer token")

    def _run_pipeline_bg(run_id: str, params: dict) -> None:
        """Run pipeline in background thread."""
        try:
            import config as _cfg
            if params.get("provider"):
                _cfg.PROVIDER = params["provider"]
            if params.get("model"):
                _cfg.MODEL = params["model"]

            from pipeline_v2 import AutomatedScientist

            progress_log: List[str] = []
            def on_progress(msg: str) -> None:
                progress_log.append(msg)
                with _RUNS_LOCK:
                    _RUNS[run_id]["progress"] = progress_log[-50:]

            with _RUNS_LOCK:
                _RUNS[run_id]["status"] = "running"
                _RUNS[run_id]["started_at"] = time.time()

            scientist = AutomatedScientist()
            results = scientist.run(
                topic=params["topic"],
                budget_usd=params.get("budget_usd", 2.0),
                max_ideation_iterations=params.get("max_iterations", 10),
                max_scientist_iterations=params.get("max_scientist_iterations", 2),
                execution_timeout=params.get("execution_timeout", 600),
                on_progress=on_progress,
                debate_enabled=params.get("debate_enabled", False),
            )

            with _RUNS_LOCK:
                _RUNS[run_id]["status"] = "completed"
                _RUNS[run_id]["results"] = results
                _RUNS[run_id]["completed_at"] = time.time()

        except Exception as exc:
            with _RUNS_LOCK:
                _RUNS[run_id]["status"] = "error"
                _RUNS[run_id]["error"] = str(exc)

    @app.get("/metrics")
    def metrics_endpoint():
        """Prometheus scrape endpoint (text/plain; version=0.0.4)."""
        from fastapi.responses import PlainTextResponse
        from observability import metrics
        return PlainTextResponse(
            metrics.render_prometheus(),
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/api/health")
    def health():
        active = sum(1 for r in _RUNS.values() if r.get("status") == "running")
        snap = health_snapshot()
        return {
            "status": "ok",
            "version": "2.0.0",
            "active_runs": active,
            "total_runs": len(_RUNS),
            "diagnostics": snap,
        }

    @app.post("/api/run", response_model=RunResponse)
    def start_run(
        req: RunRequest,
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        _require_auth(authorization)

        # 1. Rate limit.
        ip = _client_ip(request)
        ok, reason = get_rate_limiter().check(ip=ip)
        if not ok:
            raise HTTPException(429, reason)

        # 2. Validate & sanitise input (prevents prompt-injection + DoS-by-huge-topic).
        try:
            clean_topic, clean_budget, clean_iters = validate_run_input(
                req.topic, req.budget_usd, req.max_iterations, tier="pro",
            )
        except InputValidationError as exc:
            raise HTTPException(400, str(exc))

        # 3. Global concurrency cap.
        ok_c, reason_c = get_concurrency_guard().acquire(user_id=None)
        if not ok_c:
            raise HTTPException(503, reason_c)

        # 4. Reject providers with an open circuit breaker.
        provider_pref = (req.provider or "").strip().lower() or None
        if provider_pref:
            ok_cb, reason_cb = get_circuit_breaker().allow(provider_pref)
            if not ok_cb:
                get_concurrency_guard().release(user_id=None)
                raise HTTPException(503, reason_cb)

        _gc_runs()

        run_id = uuid.uuid4().hex[:12]
        params = req.dict()
        params["topic"] = clean_topic
        params["budget_usd"] = clean_budget
        params["max_iterations"] = clean_iters
        with _RUNS_LOCK:
            _RUNS[run_id] = {
                "status": "queued", "topic": clean_topic,
                "params": params, "progress": [],
                "results": None, "error": None, "created_at": time.time(),
                "client_ip": ip,
            }

        def _bg_wrapper(_rid: str, _params: dict) -> None:
            try:
                _run_pipeline_bg(_rid, _params)
            finally:
                get_concurrency_guard().release(user_id=None)

        thread = threading.Thread(target=_bg_wrapper, args=(run_id, params), daemon=True)
        thread.start()
        return RunResponse(run_id=run_id, status="queued", message=f"Pipeline started for: {clean_topic}")

    @app.get("/api/status/{run_id}", response_model=StatusResponse)
    def get_status(
        run_id: str, request: Request,
        authorization: Optional[str] = Header(None),
    ):
        _require_auth(authorization)
        ok, reason = get_rate_limiter().check(ip=_client_ip(request))
        if not ok:
            raise HTTPException(429, reason)
        with _RUNS_LOCK:
            run = _RUNS.get(run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        elapsed = 0
        if run.get("started_at"):
            elapsed = (run.get("completed_at") or time.time()) - run["started_at"]
        return StatusResponse(
            run_id=run_id, status=run["status"],
            progress=run.get("progress", [])[-20:],
            elapsed_s=round(elapsed, 1), error=run.get("error"),
        )

    @app.get("/api/results/{run_id}")
    def get_results(run_id: str):
        with _RUNS_LOCK:
            run = _RUNS.get(run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        if run["status"] != "completed":
            raise HTTPException(400, f"Run not complete yet (status: {run['status']})")
        return {"run_id": run_id, "topic": run["topic"], "results": run["results"]}

    @app.get("/api/ideas/{run_id}")
    def get_ideas(run_id: str):
        with _RUNS_LOCK:
            run = _RUNS.get(run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        if run["status"] != "completed":
            raise HTTPException(400, f"Run not complete (status: {run['status']})")
        ideas = (run.get("results") or {}).get("ideas", [])
        return {"run_id": run_id, "topic": run["topic"], "ideas_count": len(ideas), "ideas": ideas}

    @app.get("/api/runs")
    def list_runs():
        with _RUNS_LOCK:
            return [
                {"run_id": rid, "topic": r["topic"], "status": r["status"], "created_at": r.get("created_at", 0)}
                for rid, r in sorted(_RUNS.items(), key=lambda x: x[1].get("created_at", 0), reverse=True)
            ]

else:
    app = None


if __name__ == "__main__":
    if not HAS_FASTAPI:
        print("Install FastAPI: pip install fastapi uvicorn")
    else:
        import uvicorn
        print("IdeaGraph API: http://localhost:8502")
        print("API Docs: http://localhost:8502/docs")
        uvicorn.run(app, host="0.0.0.0", port=8502)
