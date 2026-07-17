"""
agents/reviewer.py - Stage 7: Automated peer review of generated papers.

Evaluates papers on NeurIPS-style criteria and provides actionable feedback.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional

from agents.base_agent import BaseAgent


class AutoReviewer(BaseAgent):
    """Automated peer reviewer for scientific papers."""

    def __init__(self):
        super().__init__(temperature=0.3)

    def review(
        self,
        paper: Dict[str, str],
        experiment_plan: Optional[Dict[str, Any]] = None,
        analysis: Optional[Dict[str, Any]] = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Review a generated paper.

        Returns dict with: scores, decision, strengths, weaknesses,
        questions, suggestions, overall_assessment.
        """
        if on_progress:
            on_progress(f"Reviewing paper: {paper.get('title', '')[:50]}")

        # Use markdown version for review (easier to parse)
        paper_text = paper.get("markdown", "")
        if not paper_text:
            paper_text = "\n\n".join(
                f"## {k.replace('_', ' ').title()}\n{v}"
                for k, v in paper.get("sections", {}).items()
            )

        # Truncate to fit context window
        paper_text = paper_text[:8000]

        system = (
            "You are an expert peer reviewer for a top-tier machine learning "
            "conference (NeurIPS/ICML level). Review the following paper critically "
            "but constructively. Evaluate on standard criteria.\n\n"
            "Return JSON with:\n"
            "- scores: {novelty: 1-10, significance: 1-10, clarity: 1-10, "
            "  soundness: 1-10, reproducibility: 1-10}\n"
            "- overall_score: 1-10 (weighted average)\n"
            "- decision: 'strong_accept' | 'accept' | 'weak_accept' | "
            "  'borderline' | 'weak_reject' | 'reject'\n"
            "- strengths: list of 3-5 specific strengths\n"
            "- weaknesses: list of 3-5 specific weaknesses\n"
            "- questions: list of 2-3 questions for authors\n"
            "- suggestions: list of 3-5 actionable improvement suggestions\n"
            "- overall_assessment: 2-3 sentence summary"
        )

        user = f"PAPER TO REVIEW:\n\n{paper_text}"

        review = self._call_json(system, user, max_tokens=2048, temperature=0.3)

        # Ensure required fields
        scores = review.get("scores", {})
        review.setdefault("scores", {
            "novelty": 5, "significance": 5, "clarity": 5,
            "soundness": 5, "reproducibility": 5,
        })
        review.setdefault("decision", "borderline")
        review.setdefault("strengths", [])
        review.setdefault("weaknesses", [])
        review.setdefault("questions", [])
        review.setdefault("suggestions", [])

        # Calculate overall score if missing
        if "overall_score" not in review:
            s = review["scores"]
            review["overall_score"] = (
                s.get("novelty", 5) * 0.25 +
                s.get("significance", 5) * 0.25 +
                s.get("clarity", 5) * 0.15 +
                s.get("soundness", 5) * 0.20 +
                s.get("reproducibility", 5) * 0.15
            )

        review["paper_title"] = paper.get("title", "")

        if on_progress:
            on_progress(
                f"Review complete: {review['decision']} "
                f"(score: {review.get('overall_score', 0):.1f}/10)"
            )

        return review

    def multi_review(
        self,
        paper: Dict[str, str],
        n_reviewers: int = 3,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run a panel of n reviewers with different perspectives,
        then produce a meta-review with final decision.
        """
        reviewer_personas = [
            ("methodologist", "You focus on experimental rigor, statistical validity, and reproducibility."),
            ("domain_expert", "You focus on novelty, positioning in the literature, and significance of contributions."),
            ("practitioner", "You focus on practical applicability, clarity of presentation, and real-world impact."),
            ("theorist", "You focus on mathematical soundness, proof correctness, and theoretical contributions."),
        ]

        reviews = []
        for i in range(min(n_reviewers, len(reviewer_personas))):
            persona_name, persona_desc = reviewer_personas[i]
            if on_progress:
                on_progress(f"  Reviewer {i+1}/{n_reviewers} ({persona_name})...")

            review = self._single_review_with_persona(paper, persona_name, persona_desc)
            review["reviewer_persona"] = persona_name
            reviews.append(review)

        # Meta-review: aggregate and decide
        if on_progress:
            on_progress("  Meta-reviewer aggregating decisions...")

        meta = self._meta_review(reviews, paper)
        meta["individual_reviews"] = reviews
        meta["paper_title"] = paper.get("title", "")

        if on_progress:
            on_progress(
                f"Multi-review complete: {meta.get('decision', '?')} "
                f"(avg score: {meta.get('overall_score', 0):.1f}/10, "
                f"{n_reviewers} reviewers)"
            )

        return meta

    def _single_review_with_persona(
        self, paper: Dict[str, str], persona_name: str, persona_desc: str,
    ) -> Dict[str, Any]:
        """Single review with a specific reviewer persona."""
        paper_text = paper.get("markdown", "")[:6000]

        system = (
            f"You are a peer reviewer with the following focus:\n{persona_desc}\n\n"
            "Review the paper critically. Return JSON with:\n"
            "- scores: {novelty: 1-10, significance: 1-10, clarity: 1-10, "
            "soundness: 1-10, reproducibility: 1-10}\n"
            "- overall_score: 1-10\n"
            "- decision: 'accept' | 'weak_accept' | 'borderline' | 'weak_reject' | 'reject'\n"
            "- strengths: list of 2-3 specific strengths\n"
            "- weaknesses: list of 2-3 specific weaknesses\n"
            "- key_suggestion: single most important improvement"
        )
        user = f"PAPER:\n{paper_text}"
        return self._call_json(system, user, max_tokens=1024, temperature=0.4)

    def _meta_review(
        self, reviews: List[Dict], paper: Dict[str, str],
    ) -> Dict[str, Any]:
        """Aggregate multiple reviews into a meta-review."""
        # Compute averages
        all_scores = [r.get("overall_score", 5) for r in reviews]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 5

        # Collect all feedback
        all_strengths = []
        all_weaknesses = []
        all_suggestions = []
        for r in reviews:
            all_strengths.extend(r.get("strengths", []))
            all_weaknesses.extend(r.get("weaknesses", []))
            if r.get("key_suggestion"):
                all_suggestions.append(r["key_suggestion"])

        # Decision based on average score
        if avg_score >= 7:
            decision = "accept"
        elif avg_score >= 6:
            decision = "weak_accept"
        elif avg_score >= 5:
            decision = "borderline"
        elif avg_score >= 4:
            decision = "weak_reject"
        else:
            decision = "reject"

        # Aggregate scores per dimension
        agg_scores = {}
        for dim in ["novelty", "significance", "clarity", "soundness", "reproducibility"]:
            vals = [(r.get("scores") or {}).get(dim, 5) for r in reviews]
            agg_scores[dim] = sum(vals) / len(vals) if vals else 5

        return {
            "overall_score": avg_score,
            "scores": agg_scores,
            "decision": decision,
            "strengths": all_strengths[:5],
            "weaknesses": all_weaknesses[:5],
            "suggestions": all_suggestions,
            "reviewer_count": len(reviews),
            "score_variance": max(all_scores) - min(all_scores) if len(all_scores) > 1 else 0,
            "consensus": "agreement" if max(all_scores) - min(all_scores) < 2 else "disagreement",
        }
