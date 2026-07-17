"""
agents/experiment_designer.py - Stage 2: Design experiments from research ideas.

Converts a research idea into a structured experiment plan with
hypothesis, variables, datasets, metrics, and baselines.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional

from agents.base_agent import BaseAgent


class ExperimentDesigner(BaseAgent):
    """Designs structured experiment plans from research ideas."""

    def __init__(self):
        super().__init__(temperature=0.4)

    def design(
        self, idea: Dict[str, Any], domain: str = "",
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Convert a research idea into a concrete experiment plan.

        Returns dict with: hypothesis, variables, datasets, metrics,
        baselines, compute_requirements, steps.
        """
        if on_progress:
            on_progress(f"Designing experiment for: {idea.get('title', '')[:50]}")

        system = (
            "You are an expert experimental scientist. Given a research idea, "
            "design a concrete, executable experiment plan. Be specific about "
            "datasets (use real, publicly available ones), metrics, and baselines. "
            "The plan must be implementable in Python with standard ML libraries "
            "(PyTorch, scikit-learn, transformers, etc.).\n\n"
            "Return JSON with these fields:\n"
            "- hypothesis: testable prediction\n"
            "- independent_vars: list of variables to manipulate\n"
            "- dependent_vars: list of variables to measure\n"
            "- controls: experimental controls\n"
            "- datasets: list of {name, source, description, size}\n"
            "- metrics: list of {name, formula_or_description}\n"
            "- baselines: list of {name, description}\n"
            "- compute_requirements: {gpu_type, estimated_hours, ram_gb}\n"
            "- steps: ordered list of experiment steps\n"
            "- expected_results: what patterns you expect to see"
        )

        user = (
            f"Research Domain: {domain or 'machine learning'}\n\n"
            f"IDEA:\n"
            f"Title: {idea.get('title', '')}\n"
            f"Method: {idea.get('method', '')}\n"
            f"Hypothesis: {idea.get('hypothesis', '')}\n"
            f"Resources: {idea.get('resources', '')}\n"
            f"Expected Outcome: {idea.get('expected_outcome', '')}\n\n"
            f"Design a complete experiment plan for this idea."
        )

        result = self._call_json(system, user, max_tokens=2048, temperature=0.4)

        # Ensure required fields
        result.setdefault("hypothesis", idea.get("hypothesis", ""))
        result.setdefault("datasets", [])
        result.setdefault("metrics", [])
        result.setdefault("baselines", [])
        result.setdefault("steps", [])
        result["idea_title"] = idea.get("title", "")

        if on_progress:
            on_progress(f"Experiment plan ready: {len(result.get('steps', []))} steps, "
                        f"{len(result.get('datasets', []))} datasets")

        return result
