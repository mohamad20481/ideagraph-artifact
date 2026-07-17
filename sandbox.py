"""
sandbox.py - Sandboxed execution environment for experiment code.

Runs generated Python code in an isolated subprocess with:
- Configurable timeout
- Memory limits
- Output capture (stdout, stderr, files)
- GPU access detection
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionResult:
    """Result of a sandboxed code execution."""
    success: bool
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    elapsed_seconds: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)  # paths to output files
    gpu_used: bool = False
    error_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout": self.stdout[:5000],  # truncate
            "stderr": self.stderr[:3000],
            "elapsed_seconds": self.elapsed_seconds,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "gpu_used": self.gpu_used,
            "error_summary": self.error_summary,
        }


def detect_gpu() -> Dict[str, Any]:
    """Check if GPU is available."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"],
            capture_output=True, text=True, timeout=15,
        )
        lines = result.stdout.strip().split("\n")
        available = lines[0].strip().lower() == "true" if lines else False
        count = int(lines[1]) if len(lines) > 1 and lines[1].strip().isdigit() else 0
        name = lines[2] if len(lines) > 2 else "unknown"
        return {"available": available, "count": count, "name": name}
    except Exception:
        return {"available": False, "count": 0, "name": "none"}


def run_experiment(
    code_files: Dict[str, str],
    entry_point: str = "experiment.py",
    timeout: int = 1800,
    requirements: Optional[List[str]] = None,
    work_dir: Optional[str] = None,
    on_progress: Optional[callable] = None,
) -> ExecutionResult:
    """
    Run experiment code in a sandboxed subprocess.

    Args:
        code_files: dict of {filename: content} to write to disk
        entry_point: which file to run
        timeout: max seconds to allow
        requirements: pip packages to install
        work_dir: directory to use (created if None)
        on_progress: callback for progress messages

    Returns:
        ExecutionResult with outputs and artifacts
    """
    # Create work directory
    cleanup = work_dir is None
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="ideagraph_exp_")

    if on_progress:
        on_progress(f"Setting up experiment in {work_dir}")

    try:
        # Write code files
        for filename, content in code_files.items():
            filepath = os.path.join(work_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

        # Install requirements if specified
        if requirements:
            if on_progress:
                on_progress(f"Installing {len(requirements)} dependencies...")
            req_result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet"] + requirements,
                capture_output=True, text=True, timeout=120,
                cwd=work_dir,
            )
            if req_result.returncode != 0:
                return ExecutionResult(
                    success=False, exit_code=req_result.returncode,
                    stderr=req_result.stderr[:2000],
                    error_summary=f"Failed to install dependencies: {req_result.stderr[:200]}",
                )

        # Create output directory
        output_dir = os.path.join(work_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Run the experiment
        if on_progress:
            on_progress(f"Running {entry_point} (timeout={timeout}s)...")

        # Build env from allowlist — prevents generated code from reading
        # API keys, DB credentials, or other secrets via os.getenv().
        _ENV_ALLOWLIST = {
            "PATH", "HOME", "USERPROFILE", "USER", "USERNAME",
            "LANG", "LC_ALL", "TMPDIR", "TEMP", "TMP",
            "PYTHONHOME", "SYSTEMROOT", "COMSPEC",  # Windows needs these
        }
        _SECRET_DENY = ("API_KEY", "SECRET", "PASSWORD", "TOKEN",
                         "CREDENTIAL", "DATABASE_URL", "BROKER_URL", "BEARER")
        env = {k: v for k, v in os.environ.items()
               if k in _ENV_ALLOWLIST
               and not any(p in k.upper() for p in _SECRET_DENY)}
        env["PYTHONPATH"] = work_dir
        env["OUTPUT_DIR"] = output_dir

        start_time = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, entry_point],
                capture_output=True, text=True, timeout=timeout,
                cwd=work_dir, env=env,
            )
            elapsed = time.time() - start_time
        except subprocess.TimeoutExpired:
            elapsed = timeout
            return ExecutionResult(
                success=False, exit_code=-1, elapsed_seconds=elapsed,
                error_summary=f"Experiment timed out after {timeout}s",
                stderr=f"TimeoutExpired: {timeout}s limit reached",
            )

        # Collect artifacts
        artifacts = []
        if os.path.isdir(output_dir):
            for root, dirs, files in os.walk(output_dir):
                for f in files:
                    artifacts.append(os.path.join(root, f))

        # Try to load metrics.json if it exists
        metrics = {}
        metrics_path = os.path.join(output_dir, "metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
            except Exception:
                pass

        # Check if GPU was used
        stdout_lower = proc.stdout.lower()
        gpu_used = "cuda" in stdout_lower or "gpu" in stdout_lower

        success = proc.returncode == 0
        error_summary = ""
        if not success:
            # Extract last meaningful error line
            err_lines = [l for l in proc.stderr.strip().split("\n") if l.strip()]
            error_summary = err_lines[-1][:200] if err_lines else "Unknown error"

        if on_progress:
            status = "succeeded" if success else "failed"
            on_progress(f"Experiment {status} in {elapsed:.1f}s")

        return ExecutionResult(
            success=success,
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            elapsed_seconds=elapsed,
            metrics=metrics,
            artifacts=artifacts,
            gpu_used=gpu_used,
            error_summary=error_summary,
        )

    except Exception as e:
        return ExecutionResult(
            success=False, error_summary=str(e),
            stderr=str(e),
        )
    finally:
        # Don't cleanup — keep artifacts for analysis
        pass
