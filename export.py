"""
export.py - Export pipeline results to PDF, DOCX, and ZIP formats.

Provides downloadable exports for:
  - Ideas report (PDF/DOCX): all generated ideas with quality scores
  - Paper export (PDF/DOCX): generated research paper
  - Code bundle (ZIP): all experiment code files
  - Full export (ZIP): everything — ideas, papers, code, analytics
"""

from __future__ import annotations

import io
import json
import os
import re
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional

# Pre-compiled patterns for Markdown → HTML conversion (export_paper_html).
_RE_BOLD = re.compile(r'\*\*(.+?)\*\*')
_RE_ITALIC = re.compile(r'\*(.+?)\*')

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, inch
    from reportlab.lib.colors import HexColor, black, white
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable,
    )
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


def export_ideas_markdown(ideas: List[Dict], topic: str, stats: Dict = None) -> str:
    """Export all ideas as a formatted Markdown report."""
    lines = [
        f"# IdeaGraph Research Report",
        f"**Topic:** {topic}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Total Ideas:** {len(ideas)}",
        "",
    ]

    if stats:
        lines.extend([
            "## Summary Statistics",
            f"- Coverage: {stats.get('coverage', 0):.1%}",
            f"- Iterations: {stats.get('iterations', '?')}",
            f"- Quality: mean={stats.get('quality_mean', 0):.3f}, max={stats.get('quality_max', 0):.3f}",
            f"- Estimated Cost: ${stats.get('estimated_cost_usd', 0):.4f}",
            "",
        ])

    sorted_ideas = sorted(ideas, key=lambda x: x.get("quality_score", 0), reverse=True)
    lines.append("## Ideas (ranked by quality)\n")

    # Per-call enum format caches: same methodology_type/novelty_level values
    # repeat across many ideas, and .replace().title() / .capitalize() were
    # being recomputed every iteration. memoize so each unique value is
    # formatted at most once per export.
    _meth_cache: Dict[str, str] = {}
    _nov_cache: Dict[str, str] = {}

    def _fmt_meth(v: Optional[str]) -> str:
        s = v or "?"
        cached = _meth_cache.get(s)
        if cached is None:
            cached = s.replace("_", " ").title()
            _meth_cache[s] = cached
        return cached

    def _fmt_nov(v: Optional[str]) -> str:
        s = v or "?"
        cached = _nov_cache.get(s)
        if cached is None:
            cached = s.capitalize()
            _nov_cache[s] = cached
        return cached

    for i, idea in enumerate(sorted_ideas, 1):
        q = idea.get("quality_score", 0)
        badge = "A+" if q >= 0.8 else "A" if q >= 0.7 else "B" if q >= 0.5 else "C" if q >= 0.3 else "D"
        lines.extend([
            f"### {i}. {idea.get('title', 'Untitled')} [{badge}]",
            f"**Quality:** {q:.3f} | "
            f"**Methodology:** {_fmt_meth(idea.get('methodology_type'))} | "
            f"**Novelty:** {_fmt_nov(idea.get('novelty_level'))} | "
            f"**Strategy:** {idea.get('source_strategy', '?')}",
            "",
            f"**Motivation:** {idea.get('motivation', 'N/A')}",
            "",
            f"**Method:** {idea.get('method', 'N/A')}",
            "",
            f"**Hypothesis:** {idea.get('hypothesis', 'N/A')}",
            "",
            f"**Resources:** {idea.get('resources', 'N/A')}",
            "",
            f"**Expected Outcome:** {idea.get('expected_outcome', 'N/A')}",
            "",
            f"**Risk Assessment:** {idea.get('risk_assessment', 'N/A')}",
            "",
            "---",
            "",
        ])

    return "\n".join(lines)


def export_ideas_html(ideas: List[Dict], topic: str, stats: Dict = None) -> str:
    """Export ideas as styled HTML (for PDF conversion)."""
    md = export_ideas_markdown(ideas, topic, stats)
    # Simple markdown → HTML conversion
    html_lines = [
        "<!DOCTYPE html><html><head>",
        "<meta charset='utf-8'>",
        "<style>",
        "body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; color: #333; line-height: 1.6; }",
        "h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }",
        "h2 { color: #2c3e50; margin-top: 30px; }",
        "h3 { color: #34495e; margin-top: 20px; }",
        "strong { color: #2c3e50; }",
        "hr { border: none; border-top: 1px solid #ecf0f1; margin: 20px 0; }",
        ".badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: bold; color: white; }",
        ".badge-a { background: #27ae60; } .badge-b { background: #f39c12; } .badge-c { background: #e74c3c; }",
        "code { background: #f8f9fa; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }",
        "</style></head><body>",
    ]

    for line in md.split("\n"):
        if line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("**") and line.endswith("**"):
            html_lines.append(f"<p><strong>{line[2:-2]}</strong></p>")
        elif line.startswith("- "):
            html_lines.append(f"<li>{line[2:]}</li>")
        elif line == "---":
            html_lines.append("<hr>")
        elif line.strip():
            # Handle bold within text (uses module-level compiled regex)
            processed = _RE_BOLD.sub(r'<strong>\1</strong>', line)
            html_lines.append(f"<p>{processed}</p>")

    html_lines.append("</body></html>")
    return "\n".join(html_lines)


def export_full_zip(
    results: Dict[str, Any],
    topic: str,
) -> bytes:
    """
    Export everything as a ZIP archive containing:
      - ideas_report.md — formatted ideas report
      - ideas_report.html — styled HTML version
      - ideas.json — raw ideas data
      - results.json — full pipeline results
      - paper.md — generated paper (if available)
      - code/ — experiment code files (if available)
      - analytics.json — stats summary
    """
    buf = io.BytesIO()

    ideas = results.get("ideas", [])
    stats = results.get("stats", {})

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Ideas report
        md_report = export_ideas_markdown(ideas, topic, stats)
        zf.writestr("ideas_report.md", md_report)

        # HTML report
        html_report = export_ideas_html(ideas, topic, stats)
        zf.writestr("ideas_report.html", html_report)

        # Raw ideas JSON
        zf.writestr("ideas.json", json.dumps(ideas, indent=2, default=str))

        # Full results (excluding large nested objects)
        safe_results = {
            "topic": topic,
            "coverage": results.get("coverage", 0),
            "stats": stats,
            "total_iterations": results.get("total_iterations", 0),
            "total_elapsed": results.get("total_elapsed", 0),
            "estimated_cost_usd": results.get("estimated_cost_usd", 0),
            "call_metrics": results.get("call_metrics", {}),
        }
        zf.writestr("results.json", json.dumps(safe_results, indent=2, default=str))

        # Paper (if v2 pipeline produced one)
        final_paper = results.get("final_paper")
        if final_paper:
            if final_paper.get("markdown"):
                zf.writestr("paper.md", final_paper["markdown"])
            if final_paper.get("latex"):
                zf.writestr("paper.tex", final_paper["latex"])

        # Review
        final_review = results.get("final_review")
        if final_review:
            zf.writestr("review.json", json.dumps(final_review, indent=2, default=str))

        # Iteration details
        iterations = results.get("iterations", [])
        if iterations:
            zf.writestr("iterations.json", json.dumps(iterations, indent=2, default=str))

        # Code files from iterations
        for iter_data in iterations:
            stages = iter_data.get("stages", {})
            code_stage = stages.get("code_generation", {})
            files = code_stage.get("files", [])
            iter_num = iter_data.get("iteration", 0)
            for fname in files:
                # We don't have the actual code content in results, just filenames
                pass

        # DAG summary
        dag_summary = results.get("dag_summary", {})
        if dag_summary:
            zf.writestr("dag_summary.json", json.dumps(dag_summary, indent=2, default=str))

        # Archive data
        archive = results.get("archive", {})
        if archive:
            zf.writestr("archive.json", json.dumps(archive, indent=2, default=str))

    buf.seek(0)
    return buf.read()


def export_paper_html(paper: Dict[str, Any], topic: str) -> str:
    """Export a generated paper as styled HTML."""
    title = paper.get("title", topic)
    md = paper.get("markdown", "")

    html = [
        "<!DOCTYPE html><html><head>",
        "<meta charset='utf-8'>",
        "<style>",
        "body { font-family: 'Times New Roman', serif; max-width: 700px; margin: 40px auto; padding: 0 20px; line-height: 1.8; color: #1a1a1a; }",
        "h1 { text-align: center; font-size: 1.5em; margin-bottom: 5px; }",
        "h2 { font-size: 1.2em; margin-top: 25px; border-bottom: 1px solid #ccc; padding-bottom: 3px; }",
        "p { text-align: justify; margin: 10px 0; }",
        ".meta { text-align: center; color: #666; font-style: italic; margin-bottom: 30px; }",
        "</style></head><body>",
        f"<h1>{title}</h1>",
        f"<p class='meta'>Generated by IdeaGraph Automated Scientist</p>",
    ]

    for line in md.split("\n"):
        if line.startswith("## "):
            html.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            html.append(f"<h1>{line[2:]}</h1>")
        elif line.strip():
            processed = _RE_BOLD.sub(r'<strong>\1</strong>', line)
            processed = _RE_ITALIC.sub(r'<em>\1</em>', processed)
            html.append(f"<p>{processed}</p>")

    html.append("</body></html>")
    return "\n".join(html)


# ============================================================================
# PDF Generation (requires reportlab)
# ============================================================================

def _build_pdf_styles():
    """Build custom PDF styles for professional reports."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "ReportTitle", parent=styles["Title"],
        fontSize=22, textColor=HexColor("#2c3e50"),
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"],
        fontSize=11, textColor=HexColor("#7f8c8d"),
        alignment=TA_CENTER, spaceAfter=20,
    ))
    styles.add(ParagraphStyle(
        "SectionHead", parent=styles["Heading2"],
        fontSize=14, textColor=HexColor("#2c3e50"),
        spaceBefore=18, spaceAfter=8,
        borderWidth=1, borderColor=HexColor("#3498db"),
        borderPadding=4,
    ))
    styles.add(ParagraphStyle(
        "IdeaTitle", parent=styles["Heading3"],
        fontSize=12, textColor=HexColor("#34495e"),
        spaceBefore=12, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "IdeaBody", parent=styles["Normal"],
        fontSize=10, leading=14,
        alignment=TA_JUSTIFY, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "MetaText", parent=styles["Normal"],
        fontSize=9, textColor=HexColor("#7f8c8d"),
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "PaperBody", parent=styles["Normal"],
        fontSize=11, leading=15,
        alignment=TA_JUSTIFY, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        "PaperTitle", parent=styles["Title"],
        fontSize=18, textColor=HexColor("#1a1a1a"),
        alignment=TA_CENTER, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "PaperSection", parent=styles["Heading2"],
        fontSize=13, textColor=HexColor("#2c3e50"),
        spaceBefore=16, spaceAfter=8,
    ))
    return styles


def _safe_text(text: str) -> str:
    """Sanitize text for reportlab Paragraph (escape XML chars)."""
    if not text:
        return ""
    text = str(text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text[:2000]  # cap length


def _quality_badge_color(q: float) -> str:
    if q >= 0.7:
        return "#27ae60"
    if q >= 0.5:
        return "#f39c12"
    if q >= 0.3:
        return "#e67e22"
    return "#e74c3c"


def generate_ideas_pdf(ideas: List[Dict], topic: str, stats: Dict = None) -> bytes:
    """
    Generate a professional PDF report of all research ideas.
    Returns PDF as bytes for st.download_button().
    """
    if not HAS_REPORTLAB:
        raise ImportError("reportlab is required for PDF export. Install: pip install reportlab")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
    )
    styles = _build_pdf_styles()
    story = []

    # ── Title page ────────────────────────────────────────────────────────
    story.append(Spacer(1, 30 * mm))
    story.append(Paragraph("IdeaGraph Research Report", styles["ReportTitle"]))
    story.append(Paragraph(
        f"Topic: {_safe_text(topic)}<br/>"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/>"
        f"Total Ideas: {len(ideas)}",
        styles["ReportSubtitle"],
    ))
    story.append(Spacer(1, 10 * mm))

    # ── Summary stats table ───────────────────────────────────────────────
    if stats:
        story.append(Paragraph("Summary Statistics", styles["SectionHead"]))
        stat_data = [
            ["Metric", "Value"],
            ["Coverage", f"{stats.get('coverage', 0):.1%}"],
            ["Iterations", str(stats.get("iterations", "?"))],
            ["Mean Quality", f"{stats.get('quality_mean', 0):.3f}"],
            ["Max Quality", f"{stats.get('quality_max', 0):.3f}"],
            ["Estimated Cost", f"${stats.get('estimated_cost_usd', 0):.4f}"],
        ]
        t = Table(stat_data, colWidths=[50 * mm, 50 * mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#3498db")),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 1), (-1, -1), HexColor("#f8f9fa")),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#ddd")),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 8 * mm))

    # ── Ideas ─────────────────────────────────────────────────────────────
    sorted_ideas = sorted(ideas, key=lambda x: x.get("quality_score", 0), reverse=True)

    story.append(Paragraph(f"Ideas ({len(ideas)} total, ranked by quality)", styles["SectionHead"]))

    for i, idea in enumerate(sorted_ideas, 1):
        q = idea.get("quality_score", 0)
        badge = "A+" if q >= 0.8 else "A" if q >= 0.7 else "B" if q >= 0.5 else "C" if q >= 0.3 else "D"
        color = _quality_badge_color(q)

        title_text = _safe_text(idea.get("title", "Untitled"))
        story.append(Paragraph(
            f'<font color="{color}"><b>[{badge}]</b></font> '
            f'<b>#{i}. {title_text}</b>',
            styles["IdeaTitle"],
        ))

        # Meta row
        mtype = _safe_text((idea.get("methodology_type") or "?").replace("_", " ").title())
        novelty = _safe_text((idea.get("novelty_level") or "?").capitalize())
        strategy = idea.get("source_strategy", "?")
        strategy_name = {"A": "Frontier", "B": "Bridge", "C": "Gap-Fill"}.get(strategy, strategy)
        story.append(Paragraph(
            f'<font color="{color}"><b>Quality: {q:.3f}</b></font> | '
            f'Methodology: {mtype} | Novelty: {novelty} | Strategy: {strategy_name}',
            styles["MetaText"],
        ))

        # Probe scores table
        scores = idea.get("probe_scores") or {}
        if scores:
            probe_data = [["Probe", "Score", "Rating"]]
            for pk, pv in scores.items():
                if isinstance(pv, (int, float)):
                    rating = "Pass" if pv >= 0.4 else "Fail"
                    probe_data.append([pk.title(), f"{pv:.2f}", rating])
            if len(probe_data) > 1:
                pt = Table(probe_data, colWidths=[35 * mm, 25 * mm, 20 * mm])
                pt.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#ecf0f1")),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#ddd")),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]))
                story.append(pt)
                story.append(Spacer(1, 2 * mm))

        # Content fields
        fields = [
            ("Motivation", idea.get("motivation", "")),
            ("Method", idea.get("method", "")),
            ("Hypothesis", idea.get("hypothesis", "")),
            ("Resources", idea.get("resources", "")),
            ("Expected Outcome", idea.get("expected_outcome", "")),
            ("Risk Assessment", idea.get("risk_assessment", "")),
        ]
        for label, value in fields:
            if value and str(value).strip():
                story.append(Paragraph(
                    f"<b>{label}:</b> {_safe_text(value)}",
                    styles["IdeaBody"],
                ))

        story.append(HRFlowable(width="100%", color=HexColor("#ecf0f1"), thickness=1, spaceAfter=4))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def export_ideas_bibtex(ideas: List[Dict], topic: str) -> str:
    """
    Export ideas as BibTeX for Zotero/Mendeley import.
    Each idea becomes a @misc entry with the full content as the abstract.
    """
    lines = ["% IdeaGraph Research Ideas — BibTeX export for Zotero/Mendeley",
             f"% Topic: {topic}",
             f"% Generated: {datetime.now().strftime('%Y-%m-%d')}",
             ""]

    for i, idea in enumerate(sorted(ideas, key=lambda x: x.get("quality_score", 0), reverse=True), 1):
        title = (idea.get("title") or "Untitled").replace("{", "").replace("}", "")
        method = idea.get("methodology_type", "research")
        novelty = idea.get("novelty_level", "moderate")
        q = idea.get("quality_score", 0)
        year = datetime.now().year

        # Build abstract from full content
        abstract_parts = []
        if idea.get("motivation"):
            abstract_parts.append(f"Motivation: {idea['motivation']}")
        if idea.get("method"):
            abstract_parts.append(f"Method: {idea['method']}")
        if idea.get("hypothesis"):
            abstract_parts.append(f"Hypothesis: {idea['hypothesis']}")
        if idea.get("expected_outcome"):
            abstract_parts.append(f"Expected: {idea['expected_outcome']}")
        abstract = " ".join(abstract_parts).replace("{", "").replace("}", "").replace("%", "\\%")

        # Generate citation key
        title_words = title.split()
        first_word = title_words[0].lower() if title_words else "idea"
        cite_key = f"ideagraph_{first_word}_{year}_{i}"
        cite_key = re.sub(r'[^a-zA-Z0-9_]', '', cite_key)

        keywords = f"{method}, {novelty}, quality={q:.2f}, ideagraph-generated"

        lines.extend([
            f"@misc{{{cite_key},",
            f"  title = {{{title}}},",
            f"  author = {{IdeaGraph AI}},",
            f"  year = {{{year}}},",
            f"  note = {{Auto-generated research idea, quality score: {q:.3f}}},",
            f"  abstract = {{{abstract[:800]}}},",
            f"  keywords = {{{keywords}}},",
            f"  howpublished = {{IdeaGraph Research Platform}},",
            "}",
            "",
        ])

    return "\n".join(lines)


def export_ideas_notion(ideas: List[Dict], topic: str, stats: Dict = None) -> str:
    """
    Export ideas as Notion-flavored Markdown with proper headers, toggles,
    and properties. Can be imported directly into Notion.
    """
    lines = [
        f"# 🔬 {topic} — Research Ideas",
        "",
        f"> Generated by **IdeaGraph** on {datetime.now().strftime('%B %d, %Y')}",
        f"> Total ideas: **{len(ideas)}**",
        "",
    ]

    if stats:
        lines.extend([
            "## 📊 Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Coverage | {stats.get('coverage', 0):.1%} |",
            f"| Iterations | {stats.get('iterations', '?')} |",
            f"| Mean Quality | {stats.get('quality_mean', 0):.3f} |",
            f"| Max Quality | {stats.get('quality_max', 0):.3f} |",
            f"| Est. Cost | ${stats.get('estimated_cost_usd', 0):.4f} |",
            "",
        ])

    sorted_ideas = sorted(ideas, key=lambda x: x.get("quality_score", 0), reverse=True)

    lines.append("## 💡 Ideas")
    lines.append("")

    for i, idea in enumerate(sorted_ideas, 1):
        q = idea.get("quality_score", 0)
        emoji = "🌟" if q >= 0.7 else "⭐" if q >= 0.5 else "✨"
        title = idea.get("title", "Untitled")
        method = (idea.get("methodology_type") or "?").replace("_", " ").title()
        novelty = (idea.get("novelty_level") or "?").capitalize()

        lines.extend([
            f"### {emoji} {i}. {title}",
            "",
            f"**Quality:** `{q:.3f}` | **Type:** `{method}` | **Novelty:** `{novelty}`",
            "",
            "<details>",
            "<summary>📖 Full details</summary>",
            "",
            f"**💡 Motivation**",
            f"{idea.get('motivation', 'N/A')}",
            "",
            f"**🔬 Method**",
            f"{idea.get('method', 'N/A')}",
            "",
            f"**🎯 Hypothesis**",
            f"{idea.get('hypothesis', 'N/A')}",
            "",
            f"**📦 Resources**",
            f"{idea.get('resources', 'N/A')}",
            "",
            f"**🎓 Expected Outcome**",
            f"{idea.get('expected_outcome', 'N/A')}",
            "",
            f"**⚠️ Risk Assessment**",
            f"{idea.get('risk_assessment', 'N/A')}",
            "",
            "</details>",
            "",
            "---",
            "",
        ])

    return "\n".join(lines)


def generate_paper_pdf(paper: Dict[str, Any], topic: str) -> bytes:
    """
    Generate a professional PDF of a research paper.
    Returns PDF as bytes for st.download_button().
    """
    if not HAS_REPORTLAB:
        raise ImportError("reportlab is required for PDF export. Install: pip install reportlab")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=25 * mm, rightMargin=25 * mm,
        topMargin=25 * mm, bottomMargin=20 * mm,
    )
    styles = _build_pdf_styles()
    story = []

    title = _safe_text(paper.get("title", topic))
    md = paper.get("markdown", "")

    # ── Title ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 15 * mm))
    story.append(Paragraph(title, styles["PaperTitle"]))
    story.append(Paragraph(
        "Generated by IdeaGraph Automated Scientist",
        styles["ReportSubtitle"],
    ))
    story.append(HRFlowable(width="100%", color=HexColor("#3498db"), thickness=2, spaceAfter=10))

    # ── Parse markdown into PDF elements ──────────────────────────────────
    for line in md.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 3 * mm))
            continue

        if line.startswith("## "):
            story.append(Paragraph(_safe_text(line[3:]), styles["PaperSection"]))
        elif line.startswith("# "):
            story.append(Paragraph(_safe_text(line[2:]), styles["PaperTitle"]))
        elif line.startswith("- ") or line.startswith("* "):
            bullet_text = _safe_text(line[2:])
            # Convert **bold** to reportlab tags
            bullet_text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', bullet_text)
            story.append(Paragraph(f"  - {bullet_text}", styles["PaperBody"]))
        else:
            safe = _safe_text(line)
            safe = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', safe)
            safe = re.sub(r'\*(.+?)\*', r'<i>\1</i>', safe)
            story.append(Paragraph(safe, styles["PaperBody"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX / Overleaf Export
# ─────────────────────────────────────────────────────────────────────────────

def _latex_escape(text: str) -> str:
    """Escape LaTeX special characters."""
    if not text:
        return ""
    for char, repl in [("\\", "\\textbackslash{}"), ("&", "\\&"),
                       ("%", "\\%"), ("$", "\\$"), ("#", "\\#"),
                       ("_", "\\_"), ("{", "\\{"), ("}", "\\}"),
                       ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")]:
        text = text.replace(char, repl)
    return text


def export_ideas_latex(ideas: List[Dict], topic: str, stats: Dict = None,
                       dag_papers: List[Dict] = None) -> str:
    """
    Export ideas as a complete, compilable LaTeX document (.tex).
    Ready for direct import into Overleaf or any LaTeX editor.
    """
    esc = _latex_escape

    def _grade(q: float) -> str:
        if q >= 0.8: return "A+"
        if q >= 0.7: return "A"
        if q >= 0.6: return "B+"
        if q >= 0.5: return "B"
        if q >= 0.4: return "C"
        return "D"

    lines = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage[margin=2.5cm]{geometry}",
        r"\usepackage{hyperref}",
        r"\usepackage{booktabs}",
        r"\usepackage{enumitem}",
        r"\usepackage{xcolor}",
        r"\definecolor{highq}{HTML}{27AE60}",
        r"\definecolor{midq}{HTML}{F39C12}",
        r"\definecolor{lowq}{HTML}{E74C3C}",
        "",
        r"\title{Research Ideas: " + esc(topic) + "}",
        r"\author{Generated by IdeaGraph}",
        r"\date{\today}",
        "",
        r"\begin{document}",
        r"\maketitle",
        "",
        r"\begin{abstract}",
        f"This document presents {len(ideas)} research ideas generated by IdeaGraph, "
        f"an AI-powered research ideation platform. ",
    ]

    if stats:
        lines.append(
            f"The pipeline achieved {stats.get('coverage', 0):.0%} coverage of the "
            f"quality-diversity archive across {stats.get('iterations', '?')} iterations, "
            f"with a mean quality score of {stats.get('quality_mean', 0):.3f}."
        )
    lines.extend([r"\end{abstract}", "", r"\tableofcontents", r"\newpage", ""])

    if dag_papers:
        lines.append(r"\section{Related Work}")
        lines.append(r"\begin{enumerate}")
        for p in dag_papers[:10]:
            lines.append(f"  \\item {esc(p.get('title', ''))} ({p.get('year', '')})")
        lines.extend([r"\end{enumerate}", ""])

    lines.extend([r"\section{Generated Research Ideas}", ""])

    for idx, idea in enumerate(ideas, 1):
        q = idea.get("quality_score", 0)
        grade = _grade(q)
        color = "highq" if q >= 0.6 else "midq" if q >= 0.4 else "lowq"
        title = esc(idea.get("title", "Untitled"))
        method_type = esc((idea.get("methodology_type") or "").replace("_", " ").title())
        novelty = esc((idea.get("novelty_level") or "").capitalize())

        lines.extend([
            f"\\subsection{{Idea {idx}: {title}}}",
            f"\\textbf{{Quality:}} \\textcolor{{{color}}}{{{grade} ({q:.2f})}} "
            f"\\quad \\textbf{{Type:}} {method_type} "
            f"\\quad \\textbf{{Novelty:}} {novelty}", "",
        ])

        for field_name, field_key in [
            ("Motivation", "motivation"), ("Method", "method"),
            ("Hypothesis", "hypothesis"), ("Resources", "resources"),
            ("Expected Outcome", "expected_outcome"),
            ("Risk Assessment", "risk_assessment"),
        ]:
            text = esc(idea.get(field_key, ""))
            if text:
                lines.extend([f"\\paragraph{{{field_name}}} {text}", ""])

        scores = idea.get("probe_scores", {})
        score_items = [(k, v) for k, v in scores.items() if isinstance(v, (int, float))]
        if score_items:
            lines.extend([
                r"\begin{table}[h]", r"\centering",
                r"\begin{tabular}{lr}", r"\toprule",
                r"\textbf{Probe} & \textbf{Score} \\", r"\midrule",
            ])
            for pk, pv in sorted(score_items, key=lambda x: x[1]):
                lines.append(f"{esc(pk.title())} & {pv:.2f} \\\\")
            lines.extend([r"\bottomrule", r"\end{tabular}",
                          f"\\caption{{Probe scores for Idea {idx}}}",
                          r"\end{table}", ""])
        lines.extend([r"\hrulefill", ""])

    if dag_papers:
        lines.append(r"\begin{thebibliography}{99}")
        for i, p in enumerate(dag_papers[:10], 1):
            authors = p.get("authors", "")
            if isinstance(authors, list):
                authors = ", ".join(esc(a.get("name", "")) for a in authors[:3])
            else:
                authors = esc(str(authors))
            lines.append(
                f"\\bibitem{{ref{i}}} {authors}. "
                f"\\textit{{{esc(p.get('title', ''))}}}. {p.get('year', '')}."
            )
        lines.append(r"\end{thebibliography}")

    lines.extend(["", r"\end{document}"])
    return "\n".join(lines)
