"""
agents/experiment_executor.py - Stage 4: Execute experiments in sandbox.

Wraps the sandbox module with retry logic and error recovery via CodeGenerator.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional

from agents.base_agent import BaseAgent
from sandbox import ExecutionResult, run_experiment


class ExperimentExecutor(BaseAgent):
    """Execute experiment code with retry and self-healing."""

    def __init__(self):
        super().__init__(temperature=0.2)

    def execute(
        self,
        code_files: Dict[str, str],
        timeout: int = 1800,
        max_retries: int = 2,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> ExecutionResult:
        """
        Execute experiment code with retries.
        On failure, attempts to fix the code and re-run.
        """
        if on_progress:
            on_progress("Starting experiment execution...")

        result = run_experiment(
            code_files=code_files,
            entry_point="experiment.py",
            timeout=timeout,
            on_progress=on_progress,
        )

        # Retry loop with self-healing
        for attempt in range(max_retries):
            if result.success:
                break

            if on_progress:
                on_progress(
                    f"Experiment failed (attempt {attempt + 1}/{max_retries}): "
                    f"{result.error_summary[:80]}"
                )

            # Try to fix the code
            fixed_code = self._fix_runtime_error(
                code_files.get("experiment.py", ""),
                result.stderr,
                result.error_summary,
            )

            if fixed_code and fixed_code != code_files.get("experiment.py", ""):
                code_files["experiment.py"] = fixed_code
                if on_progress:
                    on_progress(f"Code fixed, retrying (attempt {attempt + 2})...")

                result = run_experiment(
                    code_files=code_files,
                    entry_point="experiment.py",
                    timeout=timeout,
                    on_progress=on_progress,
                )
            else:
                if on_progress:
                    on_progress("Could not fix the error automatically.")
                break

        return result

    def _fix_runtime_error(
        self, code: str, stderr: str, error_summary: str,
    ) -> Optional[str]:
        """Use LLM to fix a runtime error in experiment code."""
        # Truncate stderr to last 30 lines for context
        stderr_lines = stderr.strip().split("\n")[-30:]
        stderr_tail = "\n".join(stderr_lines)

        system = (
            "You are a Python debugging expert. A scientific experiment script "
            "failed with the error below. Fix the code so it runs successfully. "
            "Common fixes: import missing modules, handle missing data gracefully, "
            "fix tensor shape mismatches, add try/except for network downloads, "
            "reduce batch size for OOM errors.\n\n"
            "Return ONLY the corrected Python code, no markdown fences."
        )
        user = (
            f"ERROR:\n{stderr_tail}\n\n"
            f"SUMMARY: {error_summary}\n\n"
            f"CODE:\n{code[:6000]}\n\n"
            f"Fix the error and return the complete corrected code."
        )

        try:
            fixed = self._call(system, user, max_tokens=4096, temperature=0.2)
            if fixed.startswith("```"):
                lines = fixed.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                fixed = "\n".join(lines)
            return fixed
        except Exception:
            return None
