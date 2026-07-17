"""
agents/result_analyzer.py - Stage 5: Analyze experiment results.

Processes raw outputs into structured analysis with figures and tables.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional

from agents.base_agent import BaseAgent
from sandbox import ExecutionResult


class ResultAnalyzer(BaseAgent):
    """Analyze experiment results and generate insights."""

    def __init__(self):
        super().__init__(temperature=0.3)

    def analyze(
        self,
        run_result: ExecutionResult,
        experiment_plan: Dict[str, Any],
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze experiment results into structured findings.

        Returns dict with: summary, key_findings, comparison_table,
        statistical_tests, figure_descriptions, limitations.
        """
        if on_progress:
            on_progress("Analyzing experiment results...")

        metrics = run_result.metrics
        stdout = run_result.stdout[:3000]
        artifacts = run_result.artifacts

        # Build context from results
        metrics_str = ""
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    metrics_str += f"  {k}: {v}\n"
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        metrics_str += f"  {k}.{kk}: {vv}\n"
                elif isinstance(v, list) and len(v) <= 10:
                    metrics_str += f"  {k}: {v}\n"

        system = (
            "You are an expert scientific data analyst. Analyze the experiment "
            "results and provide a structured analysis. Be precise about numbers "
            "and statistical significance. Compare against baselines.\n\n"
            "Return JSON with:\n"
            "- summary: 2-3 sentence overview of findings\n"
            "- key_findings: list of {finding, evidence, significance}\n"
            "- comparison_table: {headers: [...], rows: [[...]]}\n"
            "- statistical_notes: any statistical observations\n"
            "- supports_hypothesis: boolean + explanation\n"
            "- limitations: list of experimental limitations\n"
            "- future_work: list of suggested follow-up experiments"
        )

        user = (
            f"EXPERIMENT: {experiment_plan.get('idea_title', '')}\n"
            f"HYPOTHESIS: {experiment_plan.get('hypothesis', '')}\n\n"
            f"METRICS:\n{metrics_str or 'No structured metrics available'}\n\n"
            f"OUTPUT LOG (last 3000 chars):\n{stdout}\n\n"
            f"ARTIFACTS: {', '.join(a.split('/')[-1] for a in artifacts[:10])}\n\n"
            f"Execution: {'SUCCESS' if run_result.success else 'FAILED'} "
            f"in {run_result.elapsed_seconds:.1f}s"
            f"{' (GPU used)' if run_result.gpu_used else ''}\n\n"
            f"Provide a detailed analysis."
        )

        analysis = self._call_json(system, user, max_tokens=2048, temperature=0.3)

        # Add raw data
        analysis["raw_metrics"] = metrics
        analysis["execution_time"] = run_result.elapsed_seconds
        analysis["execution_success"] = run_result.success
        analysis["gpu_used"] = run_result.gpu_used
        analysis["artifact_count"] = len(artifacts)
        analysis["artifacts"] = [a.split("/")[-1] for a in artifacts[:20]]

        if on_progress:
            findings = len(analysis.get("key_findings", []))
            on_progress(f"Analysis complete: {findings} key findings")

        return analysis
