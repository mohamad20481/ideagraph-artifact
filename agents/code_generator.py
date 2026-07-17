"""
agents/code_generator.py - Stage 3: Generate Python experiment code.

Uses agentic approach: generate → test syntax → fix errors → retry.
Produces complete runnable experiment code from an experiment plan.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent


class CodeGenerator(BaseAgent):
    """Generates complete Python experiment code from experiment plans."""

    def __init__(self):
        super().__init__(temperature=0.3)

    def generate(
        self, experiment_plan: Dict[str, Any],
        max_retries: int = 3,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, str]:
        """
        Generate complete Python experiment code.

        Returns dict of {filename: content} for all code files.
        """
        if on_progress:
            on_progress(f"Generating code for: {experiment_plan.get('idea_title', '')[:50]}")

        # Generate the main experiment code
        code_files = {}

        # 1. Main experiment script
        code_files["experiment.py"] = self._generate_main_script(experiment_plan)

        # 2. Requirements
        code_files["requirements_exp.txt"] = self._generate_requirements(experiment_plan)

        # 3. Config
        code_files["config.yaml"] = self._generate_config(experiment_plan)

        # 4. Utils module
        code_files["utils.py"] = self._generate_utils(experiment_plan)

        # 5. Unit tests
        code_files["test_experiment.py"] = self._generate_tests(experiment_plan, code_files["experiment.py"])

        # 6. Dockerfile
        code_files["Dockerfile"] = self._generate_dockerfile(code_files["requirements_exp.txt"])

        # 7. Validate syntax and fix errors
        for attempt in range(max_retries):
            errors = self._check_syntax(code_files["experiment.py"])
            if not errors:
                break
            if on_progress:
                on_progress(f"Fixing syntax errors (attempt {attempt + 1}/{max_retries})")
            code_files["experiment.py"] = self._fix_code(
                code_files["experiment.py"], errors, experiment_plan
            )

        # 8. Quality scoring
        code_files["_quality_score"] = str(self._score_code_quality(code_files["experiment.py"]))

        if on_progress:
            on_progress(f"Code generated: {len(code_files) - 1} files (quality={code_files['_quality_score']})")

        return code_files

    def _generate_main_script(self, plan: Dict[str, Any]) -> str:
        """Generate the main experiment.py script."""
        system = (
            "You are an expert Python programmer specializing in ML experiments. "
            "Generate a COMPLETE, RUNNABLE Python script that implements the given "
            "experiment plan. The script must:\n\n"
            "1. Import all needed libraries\n"
            "2. Set random seeds for reproducibility\n"
            "3. Load or generate datasets\n"
            "4. Implement the proposed method\n"
            "5. Implement baselines for comparison\n"
            "6. Run experiments and collect metrics\n"
            "7. Save results to 'output/metrics.json'\n"
            "8. Generate plots and save to 'output/' directory\n"
            "9. Print clear progress messages\n\n"
            "Use standard libraries: torch, sklearn, numpy, matplotlib, transformers.\n"
            "Handle GPU availability gracefully (cpu fallback).\n"
            "Use small dataset sizes and few epochs for quick iteration.\n\n"
            "IMPORTANT: Output ONLY the Python code, no markdown fences."
        )

        steps_str = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan.get("steps", [])))
        datasets_str = "\n".join(
            f"  - {d.get('name', '?')}: {d.get('description', '')}"
            for d in plan.get("datasets", [])
        )
        metrics_str = "\n".join(
            f"  - {m.get('name', '?')}: {m.get('formula_or_description', '')}"
            for m in plan.get("metrics", [])
        )
        baselines_str = "\n".join(
            f"  - {b.get('name', '?')}: {b.get('description', '')}"
            for b in plan.get("baselines", [])
        )

        user = (
            f"EXPERIMENT PLAN:\n"
            f"Title: {plan.get('idea_title', '')}\n"
            f"Hypothesis: {plan.get('hypothesis', '')}\n\n"
            f"Steps:\n{steps_str}\n\n"
            f"Datasets:\n{datasets_str}\n\n"
            f"Metrics:\n{metrics_str}\n\n"
            f"Baselines:\n{baselines_str}\n\n"
            f"Generate the complete Python experiment script."
        )

        code = self._call(system, user, max_tokens=4096, temperature=0.3)

        # Clean up markdown code fences if present
        if code.startswith("```"):
            lines = code.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            code = "\n".join(lines)

        return code

    def _generate_requirements(self, plan: Dict[str, Any]) -> str:
        """Generate requirements file for the experiment."""
        # Base requirements
        reqs = [
            "torch>=2.0.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "matplotlib>=3.7.0",
            "pandas>=2.0.0",
        ]

        # Add domain-specific requirements
        plan_text = str(plan).lower()
        if "transformer" in plan_text or "nlp" in plan_text or "text" in plan_text:
            reqs.append("transformers>=4.35.0")
            reqs.append("datasets>=2.14.0")
            reqs.append("tokenizers>=0.15.0")
        if "graph" in plan_text or "gnn" in plan_text:
            reqs.append("torch-geometric>=2.4.0")
            reqs.append("networkx>=3.0")
        if "diffusion" in plan_text or "image" in plan_text:
            reqs.append("torchvision>=0.15.0")
            reqs.append("Pillow>=10.0.0")
        if "tqdm" in plan_text:
            reqs.append("tqdm>=4.65.0")

        return "\n".join(reqs)

    def _generate_config(self, plan: Dict[str, Any]) -> str:
        """Generate YAML config for the experiment."""
        compute = plan.get("compute_requirements", {})
        return (
            f"# Experiment Configuration\n"
            f"experiment:\n"
            f"  title: \"{plan.get('idea_title', 'experiment')}\"\n"
            f"  seed: 42\n"
            f"\n"
            f"training:\n"
            f"  epochs: 10\n"
            f"  batch_size: 32\n"
            f"  learning_rate: 0.001\n"
            f"  device: auto  # auto-detect GPU\n"
            f"\n"
            f"compute:\n"
            f"  gpu_type: \"{compute.get('gpu_type', 'any')}\"\n"
            f"  estimated_hours: {compute.get('estimated_hours', 1)}\n"
            f"  ram_gb: {compute.get('ram_gb', 8)}\n"
            f"\n"
            f"output:\n"
            f"  dir: output/\n"
            f"  metrics_file: output/metrics.json\n"
            f"  plots_dir: output/\n"
        )

    def _check_syntax(self, code: str) -> str:
        """Check Python code for syntax errors. Returns error string or empty."""
        try:
            compile(code, "<experiment>", "exec")
            return ""
        except SyntaxError as e:
            return f"SyntaxError at line {e.lineno}: {e.msg}"

    def _fix_code(
        self, code: str, error: str, plan: Dict[str, Any],
    ) -> str:
        """Fix code based on error message."""
        system = (
            "You are a Python debugging expert. Fix the syntax error in this code. "
            "Return ONLY the corrected Python code, no markdown fences or explanations."
        )
        user = (
            f"ERROR: {error}\n\n"
            f"CODE:\n{code}\n\n"
            f"Fix the error and return the corrected code."
        )
        fixed = self._call(system, user, max_tokens=4096, temperature=0.2)
        if fixed.startswith("```"):
            lines = fixed.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            fixed = "\n".join(lines)
        return fixed

    def _generate_utils(self, plan: Dict[str, Any]) -> str:
        """Generate utils.py with helper functions."""
        return (
            '"""utils.py - Helper functions for experiment."""\n\n'
            'import json\nimport os\nimport random\nimport numpy as np\n\n'
            'def set_seed(seed: int = 42):\n'
            '    """Set random seeds for reproducibility."""\n'
            '    random.seed(seed)\n'
            '    np.random.seed(seed)\n'
            '    try:\n'
            '        import torch\n'
            '        torch.manual_seed(seed)\n'
            '        if torch.cuda.is_available():\n'
            '            torch.cuda.manual_seed_all(seed)\n'
            '    except ImportError:\n'
            '        pass\n\n'
            'def get_device():\n'
            '    """Get best available device."""\n'
            '    try:\n'
            '        import torch\n'
            '        return torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
            '    except ImportError:\n'
            '        return "cpu"\n\n'
            'def save_metrics(metrics: dict, path: str = "output/metrics.json"):\n'
            '    """Save metrics dict to JSON."""\n'
            '    os.makedirs(os.path.dirname(path), exist_ok=True)\n'
            '    with open(path, "w") as f:\n'
            '        json.dump(metrics, f, indent=2, default=str)\n'
            '    print(f"Metrics saved to {path}")\n\n'
            'def save_plot(fig, name: str, output_dir: str = "output"):\n'
            '    """Save matplotlib figure."""\n'
            '    os.makedirs(output_dir, exist_ok=True)\n'
            '    path = os.path.join(output_dir, f"{name}.png")\n'
            '    fig.savefig(path, dpi=150, bbox_inches="tight")\n'
            '    print(f"Plot saved to {path}")\n'
        )

    def _generate_tests(self, plan: Dict[str, Any], main_code: str) -> str:
        """Generate unit tests for the experiment code."""
        system = (
            "You are a test engineer. Write Python unit tests for this experiment code. "
            "Test: imports work, data loading doesn't crash, model can do forward pass, "
            "metrics are computed correctly, output files are created.\n"
            "Use unittest. Return ONLY Python code, no markdown fences."
        )
        user = (
            f"EXPERIMENT: {plan.get('idea_title', '')}\n"
            f"CODE (first 2000 chars):\n{main_code[:2000]}\n\n"
            f"Write unit tests."
        )
        tests = self._call(system, user, max_tokens=1500, temperature=0.3)
        if tests.startswith("```"):
            lines = tests.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            tests = "\n".join(lines)
        return tests

    def _generate_dockerfile(self, requirements: str) -> str:
        """Generate Dockerfile for reproducible execution."""
        return (
            "FROM python:3.11-slim\n\n"
            "WORKDIR /experiment\n\n"
            "# Install system dependencies\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
            "    gcc g++ && rm -rf /var/lib/apt/lists/*\n\n"
            "# Install Python dependencies\n"
            "COPY requirements_exp.txt .\n"
            "RUN pip install --no-cache-dir -r requirements_exp.txt\n\n"
            "# Copy experiment code\n"
            "COPY . .\n\n"
            "# Create output directory\n"
            "RUN mkdir -p output\n\n"
            "# Run experiment\n"
            'CMD ["python", "experiment.py"]\n'
        )

    def _score_code_quality(self, code: str) -> float:
        """Score code quality on 0-1 scale."""
        score = 0.5  # base score

        # Pre-compute lowered form + line count once (was 4 separate .lower() calls)
        code_lower = code.lower()
        code_lines = code.count("\n")

        # Positive signals
        if "import" in code: score += 0.05
        if "def " in code: score += 0.05  # has functions
        if "if __name__" in code: score += 0.05  # proper entry point
        if "try:" in code: score += 0.05  # error handling
        if "seed" in code_lower: score += 0.05  # reproducibility
        if "metrics" in code_lower: score += 0.05  # saves metrics
        if "output" in code_lower: score += 0.05  # saves output
        if "print" in code: score += 0.03  # progress messages
        if "cuda" in code_lower or "device" in code_lower: score += 0.05  # GPU aware
        if code_lines > 50: score += 0.05  # substantial code

        # Negative signals
        if "TODO" in code: score -= 0.1
        if code.count("pass") > 3: score -= 0.1
        if len(code) < 200: score -= 0.2  # too short

        return max(0.0, min(1.0, score))
