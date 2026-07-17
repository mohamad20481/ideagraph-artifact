"""
acceptance_predictor.py — reviewer-aware acceptance prediction for IdeaGraph.

⚠️ Intellectual framing (read before using).

This module predicts the *probability that an idea would be accepted* at a
specific peer-review venue. It's a tool, not a recommendation. The point of
research is to find what's true, not what reviewers will accept; using this
to *select* ideas rather than to *understand* them is a misuse.

What it does:
  - Heuristic mode: a transparent linear-then-sigmoid model over the
    existing 10 probe scores, plus venue-specific compatibility terms for
    methodology_type and novelty_level. Pure Python, instant, deterministic.
  - LLM mode (optional): one call per idea where the LLM acts as a senior
    reviewer for the target venue and returns a calibrated score.

Both modes return a `AcceptanceResult` with:
  - accept_prob (0..1)
  - decision  ("accept" | "borderline" | "reject")
  - confidence
  - top_strengths and top_weaknesses (interpretable)
  - feature_contributions (so you can see *why*)

Public API:
    VENUE_PROFILES                                   → Dict[venue, profile]
    score_idea(idea, venue, mode='heuristic'|'llm', client=None)
                                                       → AcceptanceResult
    compare_venues(idea, venues=None, mode=...)      → List[AcceptanceResult]
    rank_ideas(ideas, venue, mode=...)               → List[(idea, result)]
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Venue profiles
# ─────────────────────────────────────────────────────────────────────────────
#
# Each profile is a hand-tuned model of what reviewers at that venue
# disproportionately value, calibrated against publicly reported acceptance
# rates so the bias term targets that rate at neutral (q=0.5) inputs.
#
# Weights are NOT trained on real OpenReview data (we don't ship that) —
# they're a documented prior. The proposed paper would replace these with
# fits from real (idea, venue, accept/reject) tuples; the public API is
# the same either way, so swapping in trained weights is a one-file change.

# Common feature set across all venues
_FEATURES: Tuple[str, ...] = (
    "novelty", "significance", "specificity", "clarity", "testability",
    "scalability", "risk_balance", "code", "dataset", "constraint",
    "quality_score", "novelty_match", "methodology_match", "execution_signal",
)


def _profile(tier: str, acceptance_rate: float, description: str,
              weights: Dict[str, float],
              methodology_preferences: Dict[str, float],
              novelty_preferences: Dict[str, float],
              bias: float) -> Dict[str, Any]:
    return {
        "tier": tier,
        "acceptance_rate": acceptance_rate,
        "description": description,
        "weights": weights,
        "methodology_preferences": methodology_preferences,
        "novelty_preferences": novelty_preferences,
        "bias": bias,
    }


# Default methodology preferences for several archetypes — referenced below
_TOP_TIER_METHOD_PREF = {
    "empirical_study": 0.95, "theoretical_analysis": 0.95,
    "system_design": 0.80, "dataset_creation": 0.65,
    "interdisciplinary_bridge": 0.75, "tool_library": 0.55,
    "survey_meta_analysis": 0.40,
}
_APPLIED_METHOD_PREF = {
    "empirical_study": 1.00, "system_design": 0.90,
    "dataset_creation": 0.85, "tool_library": 0.75,
    "interdisciplinary_bridge": 0.70, "theoretical_analysis": 0.55,
    "survey_meta_analysis": 0.45,
}
_NLP_METHOD_PREF = {
    "empirical_study": 1.00, "dataset_creation": 0.95,
    "system_design": 0.80, "tool_library": 0.75,
    "interdisciplinary_bridge": 0.65, "theoretical_analysis": 0.65,
    "survey_meta_analysis": 0.55,
}
_VISION_METHOD_PREF = {
    "empirical_study": 1.00, "dataset_creation": 0.85,
    "system_design": 0.85, "tool_library": 0.70,
    "interdisciplinary_bridge": 0.65, "theoretical_analysis": 0.55,
    "survey_meta_analysis": 0.45,
}
_WORKSHOP_METHOD_PREF = {
    "empirical_study": 0.95, "system_design": 0.85,
    "interdisciplinary_bridge": 0.95, "dataset_creation": 0.80,
    "tool_library": 0.85, "theoretical_analysis": 0.75,
    "survey_meta_analysis": 0.70,
}

_TOP_NOVELTY_PREF = {"incremental": 0.25, "moderate": 0.65, "substantial": 1.00}
_MID_NOVELTY_PREF = {"incremental": 0.40, "moderate": 0.85, "substantial": 1.00}
_APPLIED_NOVELTY_PREF = {"incremental": 0.50, "moderate": 0.90, "substantial": 0.95}
_WORKSHOP_NOVELTY_PREF = {"incremental": 0.75, "moderate": 0.90, "substantial": 1.00}


VENUE_PROFILES: Dict[str, Dict[str, Any]] = {
    "NeurIPS": _profile(
        tier="top", acceptance_rate=0.26,
        description=("Top-tier ML conference. Reviewers reward methodological "
                       "novelty, theoretical or large-scale empirical contributions, "
                       "and crisp specificity. Vague or incremental ideas are "
                       "filtered hard."),
        weights={
            "novelty": 1.5, "significance": 1.5, "specificity": 1.2,
            "clarity": 0.9, "testability": 0.9, "scalability": 0.8,
            "risk_balance": 0.5, "code": 0.5, "dataset": 0.4,
            "constraint": 0.3, "quality_score": 0.7,
            "novelty_match": 1.5, "methodology_match": 1.0,
            "execution_signal": 0.7,
        },
        methodology_preferences=_TOP_TIER_METHOD_PREF,
        novelty_preferences=_TOP_NOVELTY_PREF,
        bias=-7.5,
    ),
    "ICML": _profile(
        tier="top", acceptance_rate=0.27,
        description=("Top ML conference. Strong on theory + rigorous empirical. "
                       "Methodology fit and clear hypotheses matter more than "
                       "scale alone."),
        weights={
            "novelty": 1.4, "significance": 1.4, "specificity": 1.3,
            "clarity": 1.0, "testability": 1.0, "scalability": 0.7,
            "risk_balance": 0.6, "code": 0.5, "dataset": 0.4,
            "constraint": 0.3, "quality_score": 0.7,
            "novelty_match": 1.5, "methodology_match": 1.1,
            "execution_signal": 0.6,
        },
        methodology_preferences=_TOP_TIER_METHOD_PREF,
        novelty_preferences=_TOP_NOVELTY_PREF,
        bias=-7.4,
    ),
    "ICLR": _profile(
        tier="top", acceptance_rate=0.32,
        description=("Open peer review. Reviewers value clarity and reproducibility "
                       "alongside novelty. Code+experiments are standard expectation."),
        weights={
            "novelty": 1.3, "significance": 1.3, "specificity": 1.1,
            "clarity": 1.2, "testability": 1.0, "scalability": 0.7,
            "risk_balance": 0.5, "code": 0.9, "dataset": 0.6,
            "constraint": 0.4, "quality_score": 0.7,
            "novelty_match": 1.4, "methodology_match": 0.9,
            "execution_signal": 0.7,
        },
        methodology_preferences=_TOP_TIER_METHOD_PREF,
        novelty_preferences=_TOP_NOVELTY_PREF,
        bias=-6.8,
    ),
    "AAAI": _profile(
        tier="mid", acceptance_rate=0.23,
        description=("Broader AI; favors balanced contributions across theory, "
                       "applications, and systems."),
        weights={
            "novelty": 1.1, "significance": 1.2, "specificity": 1.0,
            "clarity": 1.0, "testability": 0.9, "scalability": 0.7,
            "risk_balance": 0.7, "code": 0.6, "dataset": 0.6,
            "constraint": 0.5, "quality_score": 0.8,
            "novelty_match": 1.0, "methodology_match": 0.9,
            "execution_signal": 0.6,
        },
        methodology_preferences={
            "empirical_study": 0.95, "theoretical_analysis": 0.85,
            "system_design": 0.85, "interdisciplinary_bridge": 0.75,
            "dataset_creation": 0.70, "tool_library": 0.60,
            "survey_meta_analysis": 0.55,
        },
        novelty_preferences=_MID_NOVELTY_PREF,
        bias=-5.4,
    ),
    "AISTATS": _profile(
        tier="mid", acceptance_rate=0.30,
        description=("Statistics + ML; theoretical contributions especially welcome."),
        weights={
            "novelty": 1.2, "significance": 1.1, "specificity": 1.1,
            "clarity": 1.0, "testability": 1.1, "scalability": 0.4,
            "risk_balance": 0.7, "code": 0.4, "dataset": 0.4,
            "constraint": 0.3, "quality_score": 0.7,
            "novelty_match": 1.1, "methodology_match": 1.0,
            "execution_signal": 0.5,
        },
        methodology_preferences={
            "theoretical_analysis": 1.00, "empirical_study": 0.85,
            "system_design": 0.55, "interdisciplinary_bridge": 0.70,
            "dataset_creation": 0.50, "tool_library": 0.40,
            "survey_meta_analysis": 0.45,
        },
        novelty_preferences=_MID_NOVELTY_PREF,
        bias=-5.6,
    ),
    "ACL": _profile(
        tier="top", acceptance_rate=0.22,
        description=("NLP flagship. Empirical with linguistic insight. "
                       "Datasets, evaluation rigor, and clarity carry weight."),
        weights={
            "novelty": 1.2, "significance": 1.2, "specificity": 1.1,
            "clarity": 1.2, "testability": 1.0, "scalability": 0.7,
            "risk_balance": 0.6, "code": 0.7, "dataset": 1.0,
            "constraint": 0.5, "quality_score": 0.7,
            "novelty_match": 1.3, "methodology_match": 1.0,
            "execution_signal": 0.7,
        },
        methodology_preferences=_NLP_METHOD_PREF,
        novelty_preferences=_TOP_NOVELTY_PREF,
        bias=-6.4,
    ),
    "EMNLP": _profile(
        tier="top", acceptance_rate=0.24,
        description=("Empirical NLP. Strong appetite for new datasets, careful "
                       "evaluation, and method novelty grounded in linguistics."),
        weights={
            "novelty": 1.2, "significance": 1.1, "specificity": 1.1,
            "clarity": 1.1, "testability": 1.0, "scalability": 0.7,
            "risk_balance": 0.6, "code": 0.7, "dataset": 1.1,
            "constraint": 0.5, "quality_score": 0.7,
            "novelty_match": 1.2, "methodology_match": 1.0,
            "execution_signal": 0.7,
        },
        methodology_preferences=_NLP_METHOD_PREF,
        novelty_preferences=_TOP_NOVELTY_PREF,
        bias=-6.0,
    ),
    "CVPR": _profile(
        tier="top", acceptance_rate=0.23,
        description=("Computer vision flagship. Empirical scale, benchmarks, "
                       "and clean ablations dominate."),
        weights={
            "novelty": 1.2, "significance": 1.2, "specificity": 1.1,
            "clarity": 1.0, "testability": 1.0, "scalability": 1.0,
            "risk_balance": 0.5, "code": 0.9, "dataset": 1.0,
            "constraint": 0.6, "quality_score": 0.7,
            "novelty_match": 1.2, "methodology_match": 1.0,
            "execution_signal": 0.8,
        },
        methodology_preferences=_VISION_METHOD_PREF,
        novelty_preferences=_TOP_NOVELTY_PREF,
        bias=-6.5,
    ),
    "KDD": _profile(
        tier="applied", acceptance_rate=0.22,
        description=("Applied ML / data science. Real-world impact, system "
                       "deployments, and dataset contributions are valued."),
        weights={
            "novelty": 0.9, "significance": 1.4, "specificity": 1.0,
            "clarity": 1.0, "testability": 1.0, "scalability": 1.2,
            "risk_balance": 0.7, "code": 0.9, "dataset": 1.1,
            "constraint": 0.7, "quality_score": 0.7,
            "novelty_match": 0.9, "methodology_match": 1.1,
            "execution_signal": 1.0,
        },
        methodology_preferences=_APPLIED_METHOD_PREF,
        novelty_preferences=_APPLIED_NOVELTY_PREF,
        bias=-5.5,
    ),
    "IJCAI": _profile(
        tier="mid", acceptance_rate=0.16,
        description=("Broad AI; competitive. Substantial novelty + "
                       "well-bounded contribution win."),
        weights={
            "novelty": 1.2, "significance": 1.2, "specificity": 1.0,
            "clarity": 1.0, "testability": 0.9, "scalability": 0.7,
            "risk_balance": 0.6, "code": 0.6, "dataset": 0.5,
            "constraint": 0.4, "quality_score": 0.7,
            "novelty_match": 1.1, "methodology_match": 0.9,
            "execution_signal": 0.6,
        },
        methodology_preferences={
            "empirical_study": 0.90, "theoretical_analysis": 0.85,
            "system_design": 0.80, "interdisciplinary_bridge": 0.70,
            "dataset_creation": 0.65, "tool_library": 0.55,
            "survey_meta_analysis": 0.50,
        },
        novelty_preferences=_MID_NOVELTY_PREF,
        bias=-6.6,
    ),
    "ML4H": _profile(
        tier="domain", acceptance_rate=0.40,
        description=("ML for healthcare. Clinical relevance, careful evaluation, "
                       "and reproducibility on real medical data weigh heavily."),
        weights={
            "novelty": 1.0, "significance": 1.4, "specificity": 1.1,
            "clarity": 1.1, "testability": 1.0, "scalability": 0.7,
            "risk_balance": 1.1, "code": 0.7, "dataset": 1.0,
            "constraint": 0.6, "quality_score": 0.8,
            "novelty_match": 0.9, "methodology_match": 1.0,
            "execution_signal": 0.8,
        },
        methodology_preferences={
            "empirical_study": 1.00, "dataset_creation": 0.95,
            "interdisciplinary_bridge": 0.95, "system_design": 0.80,
            "tool_library": 0.65, "theoretical_analysis": 0.55,
            "survey_meta_analysis": 0.55,
        },
        novelty_preferences=_MID_NOVELTY_PREF,
        bias=-3.6,
    ),
    "Workshop": _profile(
        tier="workshop", acceptance_rate=0.55,
        description=("Generic workshop venue. Lower bar; exploratory and "
                       "early-stage ideas welcome."),
        weights={
            "novelty": 0.7, "significance": 0.7, "specificity": 0.6,
            "clarity": 1.0, "testability": 0.7, "scalability": 0.4,
            "risk_balance": 0.4, "code": 0.4, "dataset": 0.4,
            "constraint": 0.3, "quality_score": 0.7,
            "novelty_match": 0.6, "methodology_match": 0.7,
            "execution_signal": 0.4,
        },
        methodology_preferences=_WORKSHOP_METHOD_PREF,
        novelty_preferences=_WORKSHOP_NOVELTY_PREF,
        bias=-2.4,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AcceptanceResult:
    venue: str
    accept_prob: float                    # 0..1
    decision: str                         # "accept" | "borderline" | "reject"
    confidence: float                     # 0..1
    venue_tier: str
    top_strengths: List[str] = field(default_factory=list)
    top_weaknesses: List[str] = field(default_factory=list)
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    used_llm: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "venue": self.venue, "accept_prob": self.accept_prob,
            "decision": self.decision, "confidence": self.confidence,
            "venue_tier": self.venue_tier,
            "top_strengths": list(self.top_strengths),
            "top_weaknesses": list(self.top_weaknesses),
            "feature_contributions": dict(self.feature_contributions),
            "used_llm": self.used_llm, "error": self.error,
        }

    def summary(self) -> str:
        emoji = {"accept": "✅", "borderline": "⚖️", "reject": "❌"}.get(
            self.decision, "?")
        return f"{emoji} {self.venue}: p={self.accept_prob:.2f} ({self.decision})"


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic scorer
# ─────────────────────────────────────────────────────────────────────────────

_FEATURE_LABELS: Dict[str, str] = {
    "novelty": "Novelty signal",
    "significance": "Significance / impact",
    "specificity": "Method specificity",
    "clarity": "Writing / clarity",
    "testability": "Falsifiable hypothesis",
    "scalability": "Scalability of approach",
    "risk_balance": "Risk balance",
    "code": "Implementability (code)",
    "dataset": "Data availability",
    "constraint": "Resource feasibility",
    "quality_score": "Overall idea quality",
    "novelty_match": "Novelty fit for venue",
    "methodology_match": "Methodology fit for venue",
    "execution_signal": "Tiny-experiment signal",
}


def _coerce_unit(x: Any, default: float = 0.5) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError):
        return default


def _extract_features(idea: Any, profile: Dict[str, Any]) -> Dict[str, float]:
    """Pull a [0,1] feature vector out of an Idea or dict."""
    if hasattr(idea, "to_dict"):
        d = idea.to_dict()
    elif isinstance(idea, dict):
        d = idea
    else:
        d = {}

    probes = d.get("probe_scores") or {}
    feats: Dict[str, float] = {}
    for k in ("novelty", "significance", "specificity", "clarity",
              "testability", "scalability", "risk_balance",
              "code", "dataset", "constraint"):
        feats[k] = _coerce_unit(probes.get(k, 0.5), 0.5)

    feats["quality_score"] = _coerce_unit(d.get("quality_score", 0.5), 0.5)

    # Methodology + novelty compatibility — central to venue specificity.
    # Important: when these fields are missing on the idea we look them
    # up with a sentinel so the dict's .get(...) hits the 0.5 default.
    # Defaulting to a real label like "moderate" would silently boost
    # ideas that haven't actually committed to a novelty level and break
    # the bias-calibration invariant.
    method = d.get("methodology_type")
    novelty_label = d.get("novelty_level")
    feats["methodology_match"] = _coerce_unit(
        profile["methodology_preferences"].get(method, 0.5)
        if method else 0.5,
        0.5,
    )
    feats["novelty_match"] = _coerce_unit(
        profile["novelty_preferences"].get(novelty_label, 0.5)
        if novelty_label else 0.5,
        0.5,
    )

    # Optional execution-revision signal (from execution_revisor)
    exec_sig = d.get("execution_signal")
    if exec_sig is None:
        feats["execution_signal"] = 0.5
    else:
        feats["execution_signal"] = _coerce_unit(exec_sig, 0.5)

    return feats


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _decision_from_prob(p: float) -> str:
    if p >= 0.55:
        return "accept"
    if p >= 0.30:
        return "borderline"
    return "reject"


def _confidence_from_prob(p: float) -> float:
    """High confidence near the extremes; low near the borderline."""
    return abs(p - 0.5) * 2.0


def _interpret_contributions(
    contributions: Dict[str, float],
    venue: str,
) -> Tuple[List[str], List[str]]:
    """Pick the 3 most positive and 3 most negative drivers, with labels."""
    items = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
    pos: List[str] = []
    neg: List[str] = []
    for k, v in items:
        if v > 0.10 and len(pos) < 3:
            pos.append(f"{_FEATURE_LABELS.get(k, k)} (+{v:.2f})")
    for k, v in reversed(items):
        if v < -0.10 and len(neg) < 3:
            neg.append(f"{_FEATURE_LABELS.get(k, k)} ({v:.2f})")
    if not pos and items:
        pos.append("No strong positive drivers — idea reads as "
                    "competent but unremarkable for this venue.")
    if not neg and items:
        neg.append("No strong negative drivers identified.")
    return pos, neg


def heuristic_score(idea: Any, venue: str) -> AcceptanceResult:
    """Compute acceptance probability for `idea` at `venue` using the
    venue-specific linear-then-sigmoid model. Pure Python, deterministic.
    """
    if venue not in VENUE_PROFILES:
        return AcceptanceResult(
            venue=venue, accept_prob=0.0, decision="reject",
            confidence=0.0, venue_tier="unknown",
            error=f"unknown venue '{venue}'; available: "
                  f"{sorted(VENUE_PROFILES.keys())}",
        )
    profile = VENUE_PROFILES[venue]
    feats = _extract_features(idea, profile)
    weights = profile["weights"]
    bias = profile["bias"]

    # The bias term is calibrated so that "average idea" hits roughly the
    # venue's published acceptance rate. Specifically, a midpoint vector
    # (all features = 0.5) should map to acceptance_rate.
    center_z = bias + sum(weights[k] * 0.5 for k in weights)
    target_z = math.log(profile["acceptance_rate"]
                          / (1.0 - profile["acceptance_rate"]))
    bias_correction = target_z - center_z

    z = bias + bias_correction + sum(weights[k] * feats[k] for k in weights)
    p = _sigmoid(z)

    # Per-feature contribution to z (after bias correction is divided
    # equally across features for explanatory purposes).
    contribs: Dict[str, float] = {}
    for k in weights:
        contribs[k] = weights[k] * (feats[k] - 0.5)
    pos, neg = _interpret_contributions(contribs, venue)

    return AcceptanceResult(
        venue=venue,
        accept_prob=float(p),
        decision=_decision_from_prob(p),
        confidence=_confidence_from_prob(p),
        venue_tier=profile["tier"],
        top_strengths=pos,
        top_weaknesses=neg,
        feature_contributions=contribs,
        used_llm=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optional LLM scorer (one call per idea)
# ─────────────────────────────────────────────────────────────────────────────

_LLM_SYSTEM = (
    "You are a senior reviewer at the target venue. The user gives you a "
    "research idea and a venue. Predict — calibrated against your knowledge "
    "of that venue's typical acceptance bar — the probability the idea would "
    "be accepted, and list 3 strengths and 3 weaknesses you'd write in your "
    "review. Be honest; this is a tool researchers use to understand which "
    "of their ideas align with venue preferences, not to game review. "
    "Return ONLY valid JSON with keys: accept_prob (0-1), decision "
    "('accept'|'borderline'|'reject'), confidence (0-1), strengths "
    "(list of 3 strings), weaknesses (list of 3 strings), one_line_verdict "
    "(string)."
)


def _llm_user_prompt(idea: Any, venue: str) -> str:
    if hasattr(idea, "to_dict"):
        d = idea.to_dict()
    elif isinstance(idea, dict):
        d = idea
    else:
        d = {}
    profile = VENUE_PROFILES.get(venue, {})
    title = str(d.get("title", ""))
    method = str(d.get("method", ""))[:600]
    hypothesis = str(d.get("hypothesis", ""))[:300]
    expected = str(d.get("expected_outcome", ""))[:300]
    return (
        f"Target venue: **{venue}** ({profile.get('tier', '?')} tier; "
        f"~{int(profile.get('acceptance_rate', 0.25)*100)}% acceptance rate).\n"
        f"Venue character: {profile.get('description', '')}\n\n"
        f"Idea title: {title}\n"
        f"Methodology type: {d.get('methodology_type', '?')}\n"
        f"Novelty level: {d.get('novelty_level', '?')}\n"
        f"Method: {method}\n"
        f"Hypothesis: {hypothesis}\n"
        f"Expected outcome: {expected}\n\n"
        "Return ONLY a JSON object matching the system schema."
    )


def _parse_llm_json(raw: str) -> Optional[Dict[str, Any]]:
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    start, end = s.find("{"), s.rfind("}")
    if start >= 0 and end > start:
        s = s[start:end + 1]
    try:
        return json.loads(s)
    except Exception:
        return None


def llm_score(idea: Any, venue: str, claude_client: Any) -> AcceptanceResult:
    """One LLM call: senior reviewer at `venue` judges `idea`."""
    if venue not in VENUE_PROFILES:
        return AcceptanceResult(
            venue=venue, accept_prob=0.0, decision="reject", confidence=0.0,
            venue_tier="unknown", error=f"unknown venue '{venue}'",
        )
    if claude_client is None:
        # Caller asked for LLM mode but we don't have a client → fall back
        # to the heuristic and mark used_llm=False so the UI can flag it.
        out = heuristic_score(idea, venue)
        out.error = "LLM unavailable — heuristic fallback used"
        return out
    profile = VENUE_PROFILES[venue]
    try:
        resp = claude_client.call(
            system=_LLM_SYSTEM,
            user=_llm_user_prompt(idea, venue),
            max_tokens=512, temperature=0.2,
            json_mode=True,
        )
    except Exception as e:
        out = heuristic_score(idea, venue)
        out.error = f"LLM call raised: {e!r}"
        return out
    if not getattr(resp, "success", False):
        out = heuristic_score(idea, venue)
        out.error = f"LLM call failed: {getattr(resp, 'error', '?')}"
        return out
    parsed = _parse_llm_json(getattr(resp, "text", ""))
    if not parsed:
        out = heuristic_score(idea, venue)
        out.error = "LLM response unparseable — heuristic fallback"
        return out

    p = _coerce_unit(parsed.get("accept_prob"), 0.5)
    decision = str(parsed.get("decision") or _decision_from_prob(p))
    if decision not in ("accept", "borderline", "reject"):
        decision = _decision_from_prob(p)
    conf = _coerce_unit(parsed.get("confidence"), _confidence_from_prob(p))
    strengths = parsed.get("strengths") or []
    weaknesses = parsed.get("weaknesses") or []
    if not isinstance(strengths, list):
        strengths = [str(strengths)]
    if not isinstance(weaknesses, list):
        weaknesses = [str(weaknesses)]
    return AcceptanceResult(
        venue=venue,
        accept_prob=p,
        decision=decision,
        confidence=conf,
        venue_tier=profile["tier"],
        top_strengths=[str(s)[:200] for s in strengths[:3]],
        top_weaknesses=[str(w)[:200] for w in weaknesses[:3]],
        feature_contributions={},  # LLM mode doesn't decompose
        used_llm=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public dispatchers
# ─────────────────────────────────────────────────────────────────────────────

def score_idea(idea: Any, venue: str = "NeurIPS",
                mode: str = "heuristic",
                claude_client: Any = None) -> AcceptanceResult:
    """Score one idea at one venue.

    mode:
      - 'heuristic': transparent linear-then-sigmoid model. Instant.
      - 'llm': one LLM call (requires claude_client). Falls back to
        heuristic if the client is missing or the call fails.
    """
    if mode == "llm":
        return llm_score(idea, venue, claude_client)
    return heuristic_score(idea, venue)


def compare_venues(idea: Any,
                    venues: Optional[List[str]] = None,
                    mode: str = "heuristic",
                    claude_client: Any = None) -> List[AcceptanceResult]:
    """Score one idea against several venues, sorted by accept_prob desc."""
    if not venues:
        venues = list(VENUE_PROFILES.keys())
    out: List[AcceptanceResult] = []
    for v in venues:
        out.append(score_idea(idea, v, mode=mode, claude_client=claude_client))
    out.sort(key=lambda r: r.accept_prob, reverse=True)
    return out


def rank_ideas(ideas: List[Any], venue: str = "NeurIPS",
                mode: str = "heuristic",
                claude_client: Any = None) -> List[Tuple[Any, AcceptanceResult]]:
    """Rank N ideas at one venue, sorted by accept_prob desc."""
    out: List[Tuple[Any, AcceptanceResult]] = []
    for i in ideas:
        out.append((i, score_idea(i, venue, mode=mode,
                                    claude_client=claude_client)))
    out.sort(key=lambda r: r[1].accept_prob, reverse=True)
    return out
