"""
agents/self_reflection.py - Self-reflection and quality control agent.

Reviews each pipeline stage output and provides improvement suggestions
before proceeding to the next stage. Acts as an internal quality gate.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional

from agents.base_agent import BaseAgent


class SelfReflectionAgent(BaseAgent):
    """Internal quality gate that reviews and improves stage outputs."""

    def __init__(self):
        super().__init__(temperature=0.3)

    def reflect_on_experiment(
        self, experiment_plan: Dict[str, Any], idea: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Review experiment plan and suggest improvements."""
        system = (
            "You are a meticulous research advisor reviewing an experiment plan. "
            "Identify gaps, potential confounds, missing controls, or unclear steps. "
            "Then provide a corrected/improved plan.\n\n"
            "Return JSON: {\"issues\": [\"...\"], \"severity\": \"low|medium|high\", "
            "\"improved_plan\": {same structure as input plan}, "
            "\"confidence\": 0.0-1.0}"
        )
        user = (
            f"IDEA: {idea.get('title', '')}\n"
            f"EXPERIMENT PLAN:\n"
            f"  Hypothesis: {experiment_plan.get('hypothesis', '')}\n"
            f"  Steps: {experiment_plan.get('steps', [])}\n"
            f"  Datasets: {experiment_plan.get('datasets', [])}\n"
            f"  Metrics: {experiment_plan.get('metrics', [])}\n"
            f"  Baselines: {experiment_plan.get('baselines', [])}"
        )
        return self._call_json(system, user, max_tokens=2048)

    def reflect_on_code(
        self, code: str, experiment_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Review generated code for correctness and completeness."""
        system = (
            "You are a senior ML engineer reviewing experiment code. Check for:\n"
            "1. Correctness: Does it implement the experiment plan?\n"
            "2. Completeness: Are all metrics computed and saved?\n"
            "3. Robustness: Error handling, seed setting, device handling?\n"
            "4. Output: Does it save metrics.json and plots to output/?\n\n"
            "Return JSON: {\"issues\": [\"...\"], \"severity\": \"low|medium|high\", "
            "\"fixes\": [\"description of each fix\"], \"fixed_code\": \"full corrected code\", "
            "\"confidence\": 0.0-1.0}"
        )
        user = (
            f"EXPERIMENT PLAN:\n"
            f"  Hypothesis: {experiment_plan.get('hypothesis', '')}\n"
            f"  Metrics: {experiment_plan.get('metrics', [])}\n\n"
            f"CODE:\n{code[:5000]}"
        )
        result = self._call_json(system, user, max_tokens=4096)
        # Clean markdown fences from fixed_code
        fc = result.get("fixed_code", "")
        if fc and fc.startswith("```"):
            lines = fc.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            result["fixed_code"] = "\n".join(lines)
        return result

    def reflect_on_results(
        self, analysis: Dict[str, Any], experiment_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Review analysis for completeness and validity."""
        system = (
            "You are a statistics reviewer checking experimental analysis. Verify:\n"
            "1. Are claims supported by the data?\n"
            "2. Are comparisons fair and complete?\n"
            "3. Are there missing analyses or alternative explanations?\n"
            "4. Is the hypothesis evaluation sound?\n\n"
            "Return JSON: {\"issues\": [\"...\"], \"missing_analyses\": [\"...\"], "
            "\"alternative_explanations\": [\"...\"], \"confidence\": 0.0-1.0, "
            "\"improved_summary\": \"...\"}"
        )
        user = (
            f"HYPOTHESIS: {experiment_plan.get('hypothesis', '')}\n"
            f"ANALYSIS SUMMARY: {analysis.get('summary', '')}\n"
            f"KEY FINDINGS: {analysis.get('key_findings', [])}\n"
            f"SUPPORTS HYPOTHESIS: {analysis.get('supports_hypothesis', 'unknown')}\n"
            f"RAW METRICS: {analysis.get('raw_metrics', {})}"
        )
        return self._call_json(system, user, max_tokens=1024)

    def reflect_on_paper(
        self, paper_sections: Dict[str, str],
    ) -> Dict[str, Any]:
        """Pre-review paper before sending to formal reviewer."""
        system = (
            "You are an experienced author doing a self-review before submission. "
            "Check each section for: clarity, completeness, consistency, and flow. "
            "Identify the top 3 improvements needed.\n\n"
            "Return JSON: {\"section_feedback\": {\"section_name\": \"feedback\"}, "
            "\"top_improvements\": [\"...\"], \"overall_quality\": 0.0-1.0, "
            "\"ready_for_review\": boolean}"
        )
        sections_text = "\n\n".join(
            f"## {k.replace('_', ' ').title()}\n{v[:500]}"
            for k, v in paper_sections.items()
        )
        user = f"PAPER SECTIONS:\n{sections_text}"
        return self._call_json(system, user, max_tokens=1500)
