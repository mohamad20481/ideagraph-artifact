"""
creative_lab.py - 10 creative features that make IdeaGraph feel magical.

  1. IdeaMutationLab        - 5 instant "what-if" mutations with live quality delta
  2. BlindPeerReview        - adversarial debate against your own hypothesis
  3. ResearchManifesto      - auto-generated research identity from all past ideas
  4. IdeaRoulette           - random community idea with slot-machine reveal
  5. SerendipityEngine      - cross-domain mashups you didn't ask for
  6. EvolutionReplay        - animated diff of idea versions with narration
  7. ResearchPersonality    - radar chart of your research style
  8. IdeaProphecy           - predict which ideas will trend next week
  9. IdeaOlympics           - rotating weekly leaderboard dimensions
  10. CollaboratorFinder    - "people like you" for research ideas
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from models.idea import METHODOLOGY_TYPES


# ─────────────────────────────────────────────────────────────────────────────
# Methodology display-string memoization
# ─────────────────────────────────────────────────────────────────────────────
# Methodology fields like "empirical_study" are reformatted with
# .replace("_", " ").title() at many sites — 4× inside the serendipity
# inner loop, 2× per collaborator candidate, and several rendering helpers.
# Pre-computing the small finite set once at import collapses thousands of
# tiny string-allocation calls per pipeline run into O(1) dict lookups.
_METHOD_PLAIN: Dict[str, str] = {mt: mt.replace("_", " ") for mt in METHODOLOGY_TYPES}
_METHOD_PLAIN[""] = ""
_METHOD_DISPLAY: Dict[str, str] = {k: v.title() for k, v in _METHOD_PLAIN.items()}


def _method_title(mt: Optional[str]) -> str:
    """Return Title-Cased methodology display (cached for known types)."""
    key = mt or ""
    cached = _METHOD_DISPLAY.get(key)
    if cached is not None:
        return cached
    return key.replace("_", " ").title()


def _method_plain(mt: Optional[str]) -> str:
    """Return space-separated methodology display (cached for known types)."""
    key = mt or ""
    cached = _METHOD_PLAIN.get(key)
    if cached is not None:
        return cached
    return key.replace("_", " ")


# ─────────────────────────────────────────────────────────────────────────────
# 1. IDEA MUTATION LAB
# ─────────────────────────────────────────────────────────────────────────────

MUTATION_TYPES = [
    {
        "id": "flip_hypothesis",
        "label": "Flip Hypothesis",
        "icon": "🔄",
        "description": "What if your hypothesis was the opposite?",
        "prompt": (
            "Take this research idea and FLIP the core hypothesis to its opposite. "
            "Keep the methodology and resources the same, but argue for the reversed claim. "
            "Make it scientifically plausible."
        ),
    },
    {
        "id": "swap_methodology",
        "label": "Swap Methodology",
        "icon": "🔀",
        "description": "Same problem, completely different approach.",
        "prompt": (
            "Take this research idea and REPLACE the methodology with a completely different one. "
            "If it uses empirical methods, switch to theoretical. If it uses neural networks, switch to "
            "classical methods. Keep the hypothesis and topic the same."
        ),
    },
    {
        "id": "change_dataset",
        "label": "Change Dataset",
        "icon": "📊",
        "description": "What if you used different data?",
        "prompt": (
            "Take this research idea and CHANGE the dataset/domain. "
            "If it studies molecules, switch to proteins. If it uses ImageNet, switch to medical images. "
            "Adapt the method to work with the new data."
        ),
    },
    {
        "id": "scale_down",
        "label": "Scale Down",
        "icon": "🔬",
        "description": "Make it a weekend project.",
        "prompt": (
            "Take this research idea and SIMPLIFY it drastically. "
            "Make it achievable by one person in one weekend with a laptop. "
            "Use only free, public data and simple algorithms. "
            "Keep the core insight but remove all complexity."
        ),
    },
    {
        "id": "go_wild",
        "label": "Go Wild",
        "icon": "🚀",
        "description": "What if there were no constraints?",
        "prompt": (
            "Take this research idea and make it MAXIMALLY ambitious. "
            "Assume unlimited compute, any dataset, any collaboration. "
            "What would the ideal, no-holds-barred version look like? "
            "Push the novelty to the extreme."
        ),
    },
]


def generate_mutation(idea: Dict[str, Any], mutation_type: str,
                      llm_call_fn=None) -> Optional[Dict[str, Any]]:
    """
    Generate a mutated version of an idea using the specified mutation type.
    llm_call_fn should be a function(system, user, max_tokens) -> str (JSON).
    Returns the mutated idea dict or None.
    """
    mutation = next((m for m in MUTATION_TYPES if m["id"] == mutation_type), None)
    if not mutation:
        return None

    system = (
        "You are a creative research idea mutator. "
        "Given a research idea, apply the specified mutation and return a NEW version. "
        "Return ONLY valid JSON with keys: title, motivation, method, hypothesis, "
        "resources, expected_outcome, risk_assessment, methodology_type, novelty_level."
    )
    user = (
        f"Original idea:\n"
        f"Title: {idea.get('title', '')}\n"
        f"Method: {idea.get('method', '')}\n"
        f"Hypothesis: {idea.get('hypothesis', '')}\n"
        f"Resources: {idea.get('resources', '')}\n\n"
        f"MUTATION: {mutation['prompt']}\n\n"
        f"Generate the mutated version as JSON."
    )

    if llm_call_fn:
        try:
            raw = llm_call_fn(system, user, 1024)
            if raw:
                result = json.loads(raw) if isinstance(raw, str) else raw
                result["_mutation_type"] = mutation_type
                result["_mutation_label"] = mutation["label"]
                result["_mutation_icon"] = mutation["icon"]
                result["_original_title"] = idea.get("title", "")
                return result
        except Exception:
            pass
    # Fallback: simple text manipulation (no LLM)
    return _fallback_mutation(idea, mutation_type)


def _fallback_mutation(idea: Dict, mutation_type: str) -> Dict[str, Any]:
    """Quick non-LLM mutation for instant preview."""
    mutated = dict(idea)
    if mutation_type == "flip_hypothesis":
        h = idea.get("hypothesis", "")
        mutated["hypothesis"] = f"Contrary to common belief: the opposite of '{h[:60]}' holds true."
        mutated["title"] = f"[Flipped] {idea.get('title', '')[:60]}"
    elif mutation_type == "swap_methodology":
        m = idea.get("methodology_type", "empirical_study")
        swap = {"empirical_study": "theoretical_analysis", "theoretical_analysis": "empirical_study",
                "system_design": "survey_meta_analysis", "dataset_creation": "tool_library",
                "survey_meta_analysis": "system_design", "tool_library": "dataset_creation",
                "interdisciplinary_bridge": "empirical_study"}
        mutated["methodology_type"] = swap.get(m, "empirical_study")
        mutated["title"] = f"[Swapped Method] {idea.get('title', '')[:50]}"
    elif mutation_type == "change_dataset":
        mutated["title"] = f"[New Domain] {idea.get('title', '')[:55]}"
        mutated["resources"] = "Alternative publicly available benchmark dataset"
    elif mutation_type == "scale_down":
        mutated["title"] = f"[Minimal] {idea.get('title', '')[:55]}"
        mutated["resources"] = "Single laptop, public data, 1 weekend"
        mutated["novelty_level"] = "incremental"
    elif mutation_type == "go_wild":
        mutated["title"] = f"[Moonshot] {idea.get('title', '')[:55]}"
        mutated["resources"] = "Unlimited compute, any dataset, full research team"
        mutated["novelty_level"] = "substantial"
    mutated["_mutation_type"] = mutation_type
    mutated["_mutation_label"] = mutation_type.replace("_", " ").title()
    mutated["_original_title"] = idea.get("title", "")
    return mutated


# ─────────────────────────────────────────────────────────────────────────────
# 2. BLIND PEER REVIEW
# ─────────────────────────────────────────────────────────────────────────────

def generate_blind_review(idea: Dict[str, Any], llm_call_fn=None) -> Dict[str, Any]:
    """
    Generate an adversarial peer review that argues AGAINST the idea.
    Returns {attacks: [...], questions: [...], verdict: str, strength_found: str}.
    """
    if llm_call_fn:
        system = (
            "You are a harsh but brilliant peer reviewer at a top ML conference. "
            "Your job is to DESTROY this idea by finding every weakness. "
            "Be specific, cite potential problems, and ask devastating questions. "
            "But also acknowledge ONE genuine strength.\n\n"
            "Return JSON with:\n"
            '{"attacks": ["weakness 1", "weakness 2", "weakness 3"],\n'
            ' "questions": ["adversarial question 1", "adversarial question 2", "adversarial question 3"],\n'
            ' "verdict": "one sentence brutal summary",\n'
            ' "strength_found": "one genuine strength you must acknowledge"}'
        )
        user = (
            f"IDEA TO DESTROY:\n"
            f"Title: {idea.get('title', '')}\n"
            f"Hypothesis: {idea.get('hypothesis', '')}\n"
            f"Method: {idea.get('method', '')}\n"
            f"Expected Outcome: {idea.get('expected_outcome', '')}\n\n"
            f"Find every flaw. Be merciless but scientific."
        )
        try:
            raw = llm_call_fn(system, user, 1024)
            if raw:
                return json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            pass

    # Fallback
    return {
        "attacks": [
            "The hypothesis is not clearly falsifiable.",
            "The methodology lacks a proper control group.",
            "The expected results are overly optimistic.",
        ],
        "questions": [
            "How would you handle confounding variables?",
            "What happens if your core assumption is wrong?",
            "Can you demonstrate this works on a simpler baseline first?",
        ],
        "verdict": "Interesting direction, but needs significant rigor before it's publishable.",
        "strength_found": "The core motivation identifies a real gap in the literature.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. RESEARCH MANIFESTO GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_manifesto(ideas: List[Dict[str, Any]], username: str = "Researcher",
                       llm_call_fn=None) -> Dict[str, Any]:
    """
    Analyze all past ideas and generate a research identity manifesto.
    """
    if not ideas:
        return {"manifesto": "Run some pipelines first to discover your research identity.",
                "themes": [], "style": "unknown"}

    # Extract patterns
    methods = Counter(i.get("methodology_type", "unknown") for i in ideas)
    novelties = Counter(i.get("novelty_level", "unknown") for i in ideas)
    topics = Counter()
    for i in ideas:
        for word in re.findall(r'\b[a-z]{4,}\b', (i.get("title", "") + " " + i.get("method", "")).lower()):
            if word not in {"with", "using", "based", "from", "that", "this", "which", "their", "method", "approach"}:
                topics[word] += 1

    top_method = methods.most_common(1)[0][0] if methods else "unknown"
    top_novelty = novelties.most_common(1)[0][0] if novelties else "unknown"
    top_topics = [w for w, _ in topics.most_common(8)]
    avg_quality = sum(i.get("quality_score", 0) for i in ideas) / max(len(ideas), 1)

    # Style classification
    if top_novelty == "substantial":
        style = "Risk-Taker"
    elif top_method in ("theoretical_analysis", "survey_meta_analysis"):
        style = "Deep Thinker"
    elif top_method in ("system_design", "tool_library"):
        style = "Builder"
    elif top_method == "interdisciplinary_bridge":
        style = "Cross-Pollinator"
    else:
        style = "Empiricist"

    if llm_call_fn:
        system = (
            "Generate a 2-3 sentence research manifesto for a researcher. "
            "Make it inspiring, specific, and shareable. "
            "Return JSON: {\"manifesto\": \"...\", \"tagline\": \"one-liner\"}"
        )
        user = (
            f"Researcher: {username}\n"
            f"Generated {len(ideas)} ideas\n"
            f"Top methodology: {top_method.replace('_', ' ')}\n"
            f"Preferred novelty: {top_novelty}\n"
            f"Key topics: {', '.join(top_topics[:5])}\n"
            f"Average quality: {avg_quality:.2f}\n"
            f"Research style: {style}\n\n"
            f"Write their research manifesto."
        )
        try:
            raw = llm_call_fn(system, user, 512)
            result = json.loads(raw) if isinstance(raw, str) else raw
            result["style"] = style
            result["themes"] = top_topics[:5]
            result["stats"] = {"ideas": len(ideas), "avg_quality": round(avg_quality, 2),
                               "top_method": top_method, "top_novelty": top_novelty}
            return result
        except Exception:
            pass

    # Fallback
    manifesto = (
        f"You are a {style.lower()} at the intersection of {' and '.join(top_topics[:3])}. "
        f"You favor {top_method.replace('_', ' ')} approaches with {top_novelty} novelty. "
        f"Across {len(ideas)} ideas, you consistently push for "
        f"{'ambitious breakthroughs' if top_novelty == 'substantial' else 'practical, grounded solutions'}."
    )
    return {
        "manifesto": manifesto,
        "tagline": f"{style} | {' + '.join(top_topics[:3])}",
        "style": style,
        "themes": top_topics[:5],
        "stats": {"ideas": len(ideas), "avg_quality": round(avg_quality, 2),
                  "top_method": top_method, "top_novelty": top_novelty},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. IDEA ROULETTE
# ─────────────────────────────────────────────────────────────────────────────

def spin_roulette(community_ideas: List[Dict[str, Any]],
                  exclude_titles: set = None) -> Optional[Dict[str, Any]]:
    """Pick a random high-quality community idea the user hasn't seen."""
    exclude = exclude_titles or set()
    candidates = [
        i for i in community_ideas
        if i.get("quality_score", 0) >= 0.4
        and i.get("title", "") not in exclude
    ]
    if not candidates:
        candidates = [i for i in community_ideas if i.get("title", "") not in exclude]
    if not candidates:
        return None
    return random.choice(candidates)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SERENDIPITY ENGINE (Daily Mashup)
# ─────────────────────────────────────────────────────────────────────────────

def generate_serendipity_mashups(
    user_ideas: List[Dict[str, Any]],
    community_ideas: List[Dict[str, Any]],
    n: int = 3,
) -> List[Dict[str, Any]]:
    """
    Generate N mashups by combining user's ideas with random community ideas
    from DIFFERENT domains.
    """
    if not user_ideas or not community_ideas:
        return []

    mashups = []
    used_pairs = set()

    for _ in range(n * 3):  # try more to get enough diverse pairs
        if len(mashups) >= n:
            break
        user_idea = random.choice(user_ideas)
        comm_idea = random.choice(community_ideas)

        # Skip same-domain pairs
        if user_idea.get("methodology_type") == comm_idea.get("methodology_type"):
            continue

        pair_key = (user_idea.get("title", ""), comm_idea.get("title", ""))
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)

        # Simple mashup — methodology display strings come from the memoized
        # cache (saves 4 .replace().title() calls per iteration).
        user_method = user_idea.get("methodology_type", "") or ""
        comm_method = comm_idea.get("methodology_type", "") or ""
        mashup = {
            "title": f"Mashup: {user_idea.get('title', '?')[:30]} x {comm_idea.get('title', '?')[:30]}",
            "idea_a": user_idea.get("title", ""),
            "idea_b": comm_idea.get("title", ""),
            "method_a": _method_title(user_method),
            "method_b": _method_title(comm_method),
            "synergy_hint": (
                f"Combine {_method_plain(user_method) or '?'} approach from "
                f"'{user_idea.get('title', '?')[:40]}' with the problem domain of "
                f"'{comm_idea.get('title', '?')[:40]}'"
            ),
            "estimated_novelty": "substantial",
            "quality_score": round(
                (user_idea.get("quality_score", 0.5) + comm_idea.get("quality_score", 0.5)) / 2 + 0.1,
                2,
            ),
        }
        mashups.append(mashup)

    return mashups


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVOLUTION REPLAY
# ─────────────────────────────────────────────────────────────────────────────

def build_evolution_replay(ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a step-by-step replay of how ideas evolved.
    Returns list of {version, title, changes: [{field, before, after}], quality_delta}.
    """
    # Group by lineage (parent_title chain)
    by_title = {i.get("title", ""): i for i in ideas}
    children = defaultdict(list)
    for i in ideas:
        parent = i.get("parent_title", "")
        if parent and parent in by_title:
            children[parent].append(i)

    replay = []
    for parent_title, child_list in children.items():
        parent = by_title.get(parent_title)
        if not parent:
            continue
        for child in sorted(child_list, key=lambda x: x.get("generation", 0)):
            changes = []
            for field in ["title", "method", "hypothesis", "resources", "expected_outcome"]:
                old = (parent.get(field, "") or "")[:100]
                new = (child.get(field, "") or "")[:100]
                if old != new and old and new:
                    changes.append({"field": field, "before": old, "after": new})
            q_before = parent.get("quality_score", 0)
            q_after = child.get("quality_score", 0)
            replay.append({
                "version": child.get("generation", 1),
                "parent_title": parent_title[:60],
                "child_title": child.get("title", "")[:60],
                "changes": changes,
                "quality_before": round(q_before, 3),
                "quality_after": round(q_after, 3),
                "quality_delta": round(q_after - q_before, 3),
                "narration": _narrate_evolution(changes, q_before, q_after),
            })

    return replay


def _narrate_evolution(changes: List[Dict], q_before: float, q_after: float) -> str:
    """Generate a one-sentence narration of what improved."""
    if not changes:
        return "Minor refinements applied."
    delta = q_after - q_before
    changed_fields = [c["field"] for c in changes]
    direction = "improved" if delta > 0 else "adjusted" if delta == 0 else "traded off"
    return (
        f"Quality {direction} by {abs(delta):.2f} "
        f"({q_before:.2f} -> {q_after:.2f}). "
        f"Changed: {', '.join(changed_fields)}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. RESEARCH PERSONALITY PROFILE
# ─────────────────────────────────────────────────────────────────────────────

PERSONALITY_DIMENSIONS = [
    "Empiricist",       # empirical_study ratio
    "Theorist",         # theoretical_analysis ratio
    "Builder",          # system_design + tool_library ratio
    "Risk-Taker",       # substantial novelty ratio
    "Cross-Pollinator", # interdisciplinary ratio
    "Quality-Driven",   # avg quality score
    "Prolific",         # total idea count (normalized)
    "Specific",         # avg specificity score
]


def compute_personality(ideas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute an 8-dimension research personality radar from all user ideas.
    Returns {dimensions: {name: score}, dominant: str, blind_spot: str}.
    """
    if not ideas:
        return {"dimensions": {d: 0.0 for d in PERSONALITY_DIMENSIONS},
                "dominant": "Unknown", "blind_spot": "Everything"}

    n = len(ideas)
    methods = Counter(i.get("methodology_type", "") for i in ideas)
    novelties = Counter(i.get("novelty_level", "") for i in ideas)
    avg_quality = sum(i.get("quality_score", 0) for i in ideas) / n
    avg_specificity = 0.0
    for i in ideas:
        scores = i.get("probe_scores", {})
        if isinstance(scores, dict):
            avg_specificity += scores.get("specificity", 0.5)
    avg_specificity /= max(n, 1)

    dims = {
        "Empiricist": methods.get("empirical_study", 0) / n,
        "Theorist": methods.get("theoretical_analysis", 0) / n,
        "Builder": (methods.get("system_design", 0) + methods.get("tool_library", 0)) / n,
        "Risk-Taker": novelties.get("substantial", 0) / n,
        "Cross-Pollinator": methods.get("interdisciplinary_bridge", 0) / n,
        "Quality-Driven": min(avg_quality / 0.8, 1.0),  # normalize: 0.8 = perfect
        "Prolific": min(n / 20.0, 1.0),  # normalize: 20 ideas = max
        "Specific": min(avg_specificity / 0.8, 1.0),
    }

    dominant = max(dims, key=dims.get)
    blind_spot = min(dims, key=dims.get)

    return {
        "dimensions": {k: round(v, 2) for k, v in dims.items()},
        "dominant": dominant,
        "blind_spot": blind_spot,
        "summary": f"You're primarily a {dominant}. Your blind spot is {blind_spot}.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. IDEA PROPHECY
# ─────────────────────────────────────────────────────────────────────────────

def predict_trending(user_ideas: List[Dict], community_ideas: List[Dict]) -> List[Dict[str, Any]]:
    """
    Predict which of the user's ideas are likely to trend based on
    community interest signals.
    """
    if not user_ideas:
        return []

    # Build community topic frequency
    comm_topics = Counter()
    for ci in community_ideas:
        for word in re.findall(r'\b[a-z]{4,}\b', (ci.get("title", "") + " " + ci.get("method", "")).lower()):
            comm_topics[word] += 1

    predictions = []
    for idea in user_ideas:
        idea_words = set(re.findall(
            r'\b[a-z]{4,}\b',
            (idea.get("title", "") + " " + idea.get("method", "")).lower()
        ))
        # Score = overlap with trending community topics
        overlap_score = sum(comm_topics.get(w, 0) for w in idea_words)
        quality = idea.get("quality_score", 0)
        # Combine quality + community overlap
        prophecy_score = min(0.3 * quality + 0.7 * (overlap_score / max(sum(comm_topics.values()), 1)) * 100, 1.0)
        prophecy_score = round(prophecy_score, 2)

        if prophecy_score > 0.15:
            matching_topics = [w for w in idea_words if comm_topics.get(w, 0) >= 2]
            predictions.append({
                "title": idea.get("title", ""),
                "prophecy_score": prophecy_score,
                "confidence": "high" if prophecy_score > 0.5 else "medium" if prophecy_score > 0.3 else "low",
                "reason": f"Trending topics: {', '.join(matching_topics[:4])}" if matching_topics else "Rising quality pattern",
            })

    predictions.sort(key=lambda x: x["prophecy_score"], reverse=True)
    return predictions[:5]


# ─────────────────────────────────────────────────────────────────────────────
# 9. IDEA OLYMPICS (Rotating Leaderboards)
# ─────────────────────────────────────────────────────────────────────────────

OLYMPIC_DIMENSIONS = [
    {
        "id": "highest_quality",
        "title": "Quality Champion",
        "icon": "🥇",
        "description": "Highest average idea quality this week",
        "scorer": lambda ideas: sum(i.get("quality_score", 0) for i in ideas) / max(len(ideas), 1),
    },
    {
        "id": "most_controversial",
        "title": "Most Controversial",
        "icon": "🔥",
        "description": "Highest variance in probe scores (risky but creative)",
        "scorer": lambda ideas: _avg_probe_variance(ideas),
    },
    {
        "id": "biggest_improvement",
        "title": "Biggest Improver",
        "icon": "📈",
        "description": "Largest quality gain from first to best idea",
        "scorer": lambda ideas: _quality_improvement(ideas),
    },
    {
        "id": "most_diverse",
        "title": "Methodology Explorer",
        "icon": "🌈",
        "description": "Most different methodology types used",
        "scorer": lambda ideas: len(set(i.get("methodology_type", "") for i in ideas)) / 7.0,
    },
    {
        "id": "most_prolific",
        "title": "Idea Machine",
        "icon": "⚡",
        "description": "Most ideas generated this week",
        "scorer": lambda ideas: min(len(ideas) / 20.0, 1.0),
    },
]


def _avg_probe_variance(ideas: List[Dict]) -> float:
    variances = []
    for i in ideas:
        scores = i.get("probe_scores", {})
        vals = [v for v in scores.values() if isinstance(v, (int, float))]
        if len(vals) >= 2:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            variances.append(var)
    return sum(variances) / max(len(variances), 1)


def _quality_improvement(ideas: List[Dict]) -> float:
    if not ideas:
        return 0.0
    qualities = [i.get("quality_score", 0) for i in ideas]
    return max(qualities) - min(qualities) if len(qualities) >= 2 else 0.0


def get_current_olympic_dimension() -> Dict[str, Any]:
    """Get this week's leaderboard dimension (rotates weekly)."""
    week = datetime.now().isocalendar()[1]
    idx = week % len(OLYMPIC_DIMENSIONS)
    dim = OLYMPIC_DIMENSIONS[idx]
    return {"id": dim["id"], "title": dim["title"], "icon": dim["icon"],
            "description": dim["description"]}


def compute_olympic_score(ideas: List[Dict]) -> float:
    """Compute this week's leaderboard score for a user's ideas."""
    week = datetime.now().isocalendar()[1]
    idx = week % len(OLYMPIC_DIMENSIONS)
    return round(OLYMPIC_DIMENSIONS[idx]["scorer"](ideas), 3)


# ─────────────────────────────────────────────────────────────────────────────
# 10. COLLABORATOR FINDER
# ─────────────────────────────────────────────────────────────────────────────

def find_collaborators(
    user_ideas: List[Dict[str, Any]],
    all_users: List[Dict[str, Any]],  # [{username, ideas: [...]}]
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Find users with complementary research styles.
    Complementary = different methodologies on similar topics.
    """
    if not user_ideas or not all_users:
        return []

    # User's profile
    user_methods = Counter(i.get("methodology_type", "") for i in user_ideas)
    user_topics = set()
    for i in user_ideas:
        for w in re.findall(r'\b[a-z]{4,}\b', (i.get("title", "") + " " + i.get("method", "")).lower()):
            user_topics.add(w)

    results = []
    for other in all_users:
        other_ideas = other.get("ideas", [])
        if not other_ideas:
            continue

        other_methods = Counter(i.get("methodology_type", "") for i in other_ideas)
        other_topics = set()
        for i in other_ideas:
            for w in re.findall(r'\b[a-z]{4,}\b', (i.get("title", "") + " " + i.get("method", "")).lower()):
                other_topics.add(w)

        # Topic overlap (similar interests)
        topic_overlap = len(user_topics & other_topics) / max(len(user_topics | other_topics), 1)

        # Method difference (complementary skills)
        all_methods = set(user_methods.keys()) | set(other_methods.keys())
        shared_methods = set(user_methods.keys()) & set(other_methods.keys())
        method_diff = 1.0 - (len(shared_methods) / max(len(all_methods), 1))

        # Score: high topic overlap + high method difference = great collaborator
        collab_score = 0.5 * topic_overlap + 0.5 * method_diff

        if collab_score > 0.2:
            # Find complementary strengths
            user_strong = user_methods.most_common(1)[0][0] if user_methods else "?"
            other_strong = other_methods.most_common(1)[0][0] if other_methods else "?"
            shared_topics = sorted(user_topics & other_topics)[:4]

            # Pull memoized display strings once instead of 4× .replace/.title chains
            user_strong_plain = _method_plain(user_strong)
            other_strong_plain = _method_plain(other_strong)
            results.append({
                "username": other.get("username", "Unknown"),
                "score": round(collab_score, 2),
                "their_strength": _method_title(other_strong),
                "your_strength": _method_title(user_strong),
                "shared_topics": shared_topics,
                "reason": (
                    f"You do {user_strong_plain}, they do {other_strong_plain}. "
                    f"Shared interests: {', '.join(shared_topics[:3]) if shared_topics else 'related fields'}."
                ),
                "ideas_count": len(other_ideas),
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]
