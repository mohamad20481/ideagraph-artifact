"""
agents/paper_writer.py - Stage 6: Generate full LaTeX scientific papers.

Produces complete academic papers with 8 sections, figures, tables, and references.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional

from agents.base_agent import BaseAgent


class PaperWriter(BaseAgent):
    """Generate full LaTeX scientific papers from research artifacts."""

    SECTIONS = [
        "abstract", "introduction", "related_work", "method",
        "experimental_setup", "results", "discussion", "conclusion",
    ]

    def __init__(self):
        super().__init__(temperature=0.4)

    def write(
        self,
        idea: Dict[str, Any],
        experiment_plan: Dict[str, Any],
        analysis: Dict[str, Any],
        dag_summary: Optional[Dict[str, Any]] = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, str]:
        """
        Generate a full paper.

        Returns dict with:
        - "latex": full LaTeX source
        - "markdown": markdown version
        - "title": paper title
        - "sections": dict of {section_name: content}
        """
        if on_progress:
            on_progress(f"Writing paper: {idea.get('title', '')[:50]}")

        sections = {}
        for section_name in self.SECTIONS:
            if on_progress:
                on_progress(f"  Writing section: {section_name}...")
            sections[section_name] = self._write_section(
                section_name, idea, experiment_plan, analysis, dag_summary
            )

        # Build references from DAG
        references = self._build_references(dag_summary)

        # Compile full LaTeX
        title = idea.get("title", "Untitled Research Paper")
        latex = self._compile_latex(title, sections, references)
        markdown = self._compile_markdown(title, sections)

        if on_progress:
            on_progress(f"Paper complete: {len(latex)} chars LaTeX, 8 sections")

        return {
            "latex": latex,
            "markdown": markdown,
            "title": title,
            "sections": sections,
        }

    def _write_section(
        self, section_name: str, idea: Dict, plan: Dict,
        analysis: Dict, dag: Optional[Dict],
    ) -> str:
        """Write a single paper section."""
        context = self._build_section_context(section_name, idea, plan, analysis, dag)

        section_instructions = {
            "abstract": "Write a concise abstract (150-250 words) summarizing the problem, method, key results, and conclusions.",
            "introduction": "Write the introduction: motivate the problem, state contributions (3-4 bullet points), and outline the paper structure.",
            "related_work": "Write the related work section: discuss relevant prior work, identify gaps, and position this work's contribution.",
            "method": "Write the method section: describe the proposed approach in detail with mathematical notation where appropriate.",
            "experimental_setup": "Write the experimental setup: datasets, evaluation metrics, baselines, implementation details, hyperparameters.",
            "results": "Write the results section: present quantitative results, comparison tables, and key observations from figures.",
            "discussion": "Write the discussion: interpret results, explain why the method works, address limitations, and suggest future work.",
            "conclusion": "Write the conclusion: summarize main contributions and findings in 1-2 paragraphs.",
        }

        system = (
            "You are an expert scientific paper writer. Write the specified section "
            "of an academic paper. Use formal academic prose. Be precise and concise. "
            "Use LaTeX formatting for math ($equation$). Reference figures and tables "
            "where appropriate (Figure 1, Table 1, etc.).\n\n"
            "Output the section content directly (no section header)."
        )

        user = (
            f"SECTION: {section_name.replace('_', ' ').title()}\n"
            f"INSTRUCTIONS: {section_instructions.get(section_name, 'Write this section.')}\n\n"
            f"{context}"
        )

        return self._call(system, user, max_tokens=1500, temperature=0.4)

    def _build_section_context(
        self, section: str, idea: Dict, plan: Dict,
        analysis: Dict, dag: Optional[Dict],
    ) -> str:
        """Build context string relevant to the section being written."""
        base = (
            f"TITLE: {idea.get('title', '')}\n"
            f"METHOD: {idea.get('method', '')}\n"
            f"HYPOTHESIS: {plan.get('hypothesis', idea.get('hypothesis', ''))}\n"
        )

        if section in ("abstract", "introduction", "conclusion"):
            findings = analysis.get("key_findings", [])
            findings_str = "\n".join(
                f"  - {f.get('finding', '')}" for f in findings[:5]
            ) if findings else "No specific findings available"
            base += (
                f"MOTIVATION: {idea.get('motivation', '')}\n"
                f"KEY FINDINGS:\n{findings_str}\n"
                f"SUMMARY: {analysis.get('summary', '')}\n"
            )

        if section == "related_work" and dag:
            papers = dag.get("papers", [])[:10]
            refs = "\n".join(
                f"  - {p.get('title', '')} ({p.get('year', '?')})"
                for p in papers
            )
            base += f"RELATED PAPERS:\n{refs}\n"

        if section in ("method", "experimental_setup"):
            steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan.get("steps", [])))
            datasets = "\n".join(
                f"  - {d.get('name', '')}: {d.get('description', '')}"
                for d in plan.get("datasets", [])
            )
            base += f"STEPS:\n{steps}\nDATASETS:\n{datasets}\n"

        if section in ("results", "discussion"):
            metrics = analysis.get("raw_metrics", {})
            metrics_str = "\n".join(f"  {k}: {v}" for k, v in metrics.items() if isinstance(v, (int, float)))
            base += (
                f"METRICS:\n{metrics_str}\n"
                f"SUPPORTS HYPOTHESIS: {analysis.get('supports_hypothesis', 'unknown')}\n"
                f"LIMITATIONS: {', '.join(analysis.get('limitations', []))}\n"
            )

        return base

    def _build_references(self, dag: Optional[Dict]) -> str:
        """Build BibTeX references from DAG papers."""
        if not dag:
            return ""
        refs = []
        for i, p in enumerate(dag.get("papers", [])[:15]):
            pid = p.get("id", f"ref{i}")[:20]
            title = p.get("title", "Untitled")
            year = p.get("year", 2024)
            refs.append(
                f"@article{{{pid},\n"
                f"  title={{{title}}},\n"
                f"  year={{{year}}},\n"
                f"}}"
            )
        return "\n\n".join(refs)

    def _compile_latex(self, title: str, sections: Dict[str, str], references: str) -> str:
        """Compile sections into a full LaTeX document."""
        section_tex = ""
        for name in self.SECTIONS:
            heading = name.replace("_", " ").title()
            if name == "abstract":
                section_tex += f"\\begin{{abstract}}\n{sections.get(name, '')}\n\\end{{abstract}}\n\n"
            else:
                section_tex += f"\\section{{{heading}}}\n{sections.get(name, '')}\n\n"

        return (
            "\\documentclass{article}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage{amsmath,amssymb}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{hyperref}\n"
            "\\usepackage[margin=1in]{geometry}\n\n"
            f"\\title{{{title}}}\n"
            "\\author{IdeaGraph Automated Scientist}\n"
            "\\date{\\today}\n\n"
            "\\begin{document}\n"
            "\\maketitle\n\n"
            f"{section_tex}"
            "\\bibliographystyle{plain}\n"
            "\\end{document}\n"
        )

    def _compile_markdown(self, title: str, sections: Dict[str, str]) -> str:
        """Compile sections into markdown."""
        md = f"# {title}\n\n*Generated by IdeaGraph Automated Scientist*\n\n"
        for name in self.SECTIONS:
            heading = name.replace("_", " ").title()
            md += f"## {heading}\n\n{sections.get(name, '')}\n\n"
        return md
