"""
idea_enhancer.py - 5 enhancements to make idea generation dramatically better.

  1. IdeationKnobs        - user-controllable creativity, time, risk levers
  2. ReproducibilityEnforcer - structured reproducibility specs per idea
  3. DomainPersonaTuner   - domain-aware expert personas
  4. AdversarialGenerator - generate the contrary version of any idea
  5. FMEAGenerator        - structured failure-mode analysis per idea

Each enhancement is a pure function so it can be wired into the existing
ideation pipeline without disrupting the agent classes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. IDEATION KNOBS — user-controllable generation parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IdeationKnobs:
    """User-controllable levers for idea generation style."""
    creativity_level: float = 0.7      # 0.0 = safe, 1.0 = radical
    time_budget_weeks: int = 12        # 2 = hackathon, 12 = paper, 52 = thesis
    risk_tolerance: str = "medium"     # "low", "medium", "high"
    enable_adversarial: bool = False   # generate contrary twin?
    enable_fmea: bool = True           # require structured failure analysis?
    enable_reproducibility: bool = True  # require concrete specs?
    domain_persona: str = "auto"       # "auto", "ml", "nlp", "vision", "rl", "bio", ...

    def temperature(self) -> float:
        """Map creativity_level to LLM temperature (0.4 - 1.0)."""
        return round(0.4 + 0.6 * max(0.0, min(1.0, self.creativity_level)), 2)

    def time_phase_count(self) -> int:
        """Number of phases the idea should decompose into."""
        if self.time_budget_weeks <= 4:
            return 2
        if self.time_budget_weeks <= 16:
            return 3
        return 4

    def risk_descriptor(self) -> str:
        """One-line description of risk tolerance for prompt injection."""
        return {
            "low":    "favor proven, low-risk approaches with clear precedent",
            "medium": "balance novelty and risk; prefer ideas with at least one published precedent",
            "high":   "embrace high-risk, high-reward ideas even without prior precedent",
        }.get(self.risk_tolerance, "balance novelty and risk")

    def to_prompt_context(self) -> str:
        """Compose a prompt-injectable description of the knobs."""
        return (
            f"\n\nGENERATION CONSTRAINTS:\n"
            f"- Creativity level: {self.creativity_level:.0%} "
            f"({'safe/incremental' if self.creativity_level < 0.4 else 'balanced' if self.creativity_level < 0.75 else 'radical/moonshot'})\n"
            f"- Time budget: {self.time_budget_weeks} weeks total\n"
            f"- Risk tolerance: {self.risk_descriptor()}\n"
            f"- Decompose into {self.time_phase_count()} phases with weekly milestones."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. REPRODUCIBILITY ENFORCER
# ─────────────────────────────────────────────────────────────────────────────

REPRODUCIBILITY_PROMPT_INJECT = """

REPRODUCIBILITY REQUIREMENTS (mandatory in your response):
The 'resources' field MUST include all of:
1. Exact framework versions (e.g., "PyTorch 2.0.1 + CUDA 11.8", NOT "PyTorch 2.x")
2. Specific dataset name + size (e.g., "ImageNet-1K, 138GB", NOT "standard benchmark")
3. Exact GPU type + estimated GPU-hours (e.g., "8x A100-40GB, ~336 GPU-hours", NOT "GPU cluster")
4. Random seed for determinism (e.g., "seed=42")
5. Hyperparameter ranges (e.g., "lr in [1e-5, 1e-3], batch=[32,64,128]")
6. Expected runtime in wall-clock hours
"""

# Patterns that suggest vague/non-reproducible specs
_VAGUE_PATTERNS = [
    r"\bstandard (?:benchmark|dataset|library)\b",
    r"\bvarious (?:methods|approaches|datasets)\b",
    r"\bappropriate (?:hyperparameters|settings)\b",
    r"\b(?:moderate|sufficient|reasonable) (?:compute|resources)\b",
    r"\b(?:GPU|TPU|cluster) (?:as needed|when available)\b",
    r"\b(?:python|pytorch|tensorflow) [a-z\.]+\b(?!.*\d)",  # "PyTorch 2.x" w/o version
]


def reproducibility_score(idea: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a reproducibility score (0.0-1.0) based on concrete specs in
    the idea's resources/method fields. Returns score + missing items.
    """
    text = (idea.get("resources", "") + " " + idea.get("method", "")).lower()

    # Check each dimension
    has_version = bool(re.search(r"\b(?:pytorch|tensorflow|jax|sklearn|numpy|cuda)\s*[v\d]+\.[\d.]+", text))
    has_dataset_size = bool(re.search(r"\d+\s*(?:gb|mb|tb|k|m|million|thousand)", text))
    has_gpu_hours = bool(re.search(r"\d+\s*(?:gpu[- ]?hours?|hours?\s*on|days?\s*on)", text))
    has_seed = "seed" in text or "random_state" in text
    has_hyperparams = bool(re.search(r"(?:lr|learning rate|batch[ _]size|epochs?)\s*[=:]\s*[\d\[]", text))
    has_gpu_type = bool(re.search(r"\b(?:a100|v100|h100|t4|rtx|gtx|tpu|m1|m2)\b", text))

    checks = {
        "framework_versions": has_version,
        "dataset_size": has_dataset_size,
        "gpu_hours": has_gpu_hours,
        "random_seed": has_seed,
        "hyperparameters": has_hyperparams,
        "gpu_type": has_gpu_type,
    }
    score = sum(checks.values()) / len(checks)

    # Penalty for vague language
    vague_count = sum(1 for p in _VAGUE_PATTERNS if re.search(p, text))
    score = max(0.0, score - 0.1 * vague_count)

    missing = [k.replace("_", " ") for k, v in checks.items() if not v]
    return {
        "score": round(score, 2),
        "checks": checks,
        "missing": missing,
        "vague_phrases": vague_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. DOMAIN PERSONA TUNER
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_PERSONAS = {
    "ml": {
        "name": "ML Researcher",
        "focus": "sample efficiency, generalization, theoretical guarantees",
        "concerns": "data leakage, distribution shift, fair comparison to baselines",
        "metrics": "accuracy, F1, AUROC, perplexity, FLOPs",
    },
    "nlp": {
        "name": "NLP Researcher",
        "focus": "tokenization, alignment, multilingual transfer, prompt design",
        "concerns": "data contamination from pre-training, evaluation on multiple languages, hallucination",
        "metrics": "BLEU, ROUGE, exact-match accuracy, MMLU score, BERTScore",
    },
    "vision": {
        "name": "Computer Vision Researcher",
        "focus": "data augmentation, architecture design, robustness to occlusion/lighting",
        "concerns": "ImageNet bias, OOD generalization, adversarial robustness",
        "metrics": "top-1/top-5 accuracy, mAP, IoU, FID, FPS",
    },
    "rl": {
        "name": "Reinforcement Learning Researcher",
        "focus": "sample efficiency, exploration-exploitation, credit assignment, sim-to-real",
        "concerns": "reward hacking, environment overfitting, instability under function approximation",
        "metrics": "episodic return, sample complexity, wall-clock time, success rate",
    },
    "bio": {
        "name": "Computational Biology Researcher",
        "focus": "biological plausibility, interpretability, integration with wet-lab data",
        "concerns": "batch effects, small sample sizes, regulatory approval pathways",
        "metrics": "AUROC for prediction, MCC, fold-change, p-values with multiple-testing correction",
    },
    "graph": {
        "name": "Graph ML Researcher",
        "focus": "permutation invariance, scalability to billion-node graphs, expressiveness",
        "concerns": "over-smoothing, heterophily, train/test split leakage in transductive settings",
        "metrics": "AUC, accuracy, precision@k, hits@10, MRR",
    },
    "drug": {
        "name": "Drug Discovery Researcher",
        "focus": "ADMET properties, target identification, synthetic accessibility",
        "concerns": "scaffold hopping, IP space, hit-to-lead optimization",
        "metrics": "QED, SAS, binding affinity (Kd, IC50), Tanimoto similarity",
    },
    "robotics": {
        "name": "Robotics Researcher",
        "focus": "real-world robustness, sim-to-real transfer, safety constraints",
        "concerns": "sensor noise, action delay, partial observability, hardware wear",
        "metrics": "success rate on physical hardware, time-to-task-completion, safety violations",
    },
}


def detect_domain(topic: str) -> str:
    """Auto-detect research domain from the topic string."""
    t = topic.lower()
    if any(w in t for w in ["nlp", "language", "translation", "llm", "text", "tokeniz"]):
        return "nlp"
    if any(w in t for w in ["vision", "image", "object detection", "segmentation", "video"]):
        return "vision"
    if any(w in t for w in ["reinforcement", "rl ", "policy gradient", "agent", "multi-agent"]):
        return "rl"
    if any(w in t for w in ["protein", "genom", "biolog", "cell", "rna", "dna", "molec"]):
        return "bio"
    if any(w in t for w in ["graph neural", "gnn", "knowledge graph", "node classification"]):
        return "graph"
    if any(w in t for w in ["drug", "pharma", "compound", "ligand", "binding"]):
        return "drug"
    if any(w in t for w in ["robot", "manipulation", "grasping", "navigation"]):
        return "robotics"
    return "ml"


def domain_persona_prompt(domain: str = "auto", topic: str = "") -> str:
    """Generate a domain-specific persona system prompt segment."""
    if domain == "auto":
        domain = detect_domain(topic)
    persona = DOMAIN_PERSONAS.get(domain, DOMAIN_PERSONAS["ml"])
    return (
        f"\n\nYou are an expert {persona['name']}. "
        f"Focus on: {persona['focus']}. "
        f"Be especially mindful of: {persona['concerns']}. "
        f"Use domain-standard metrics: {persona['metrics']}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. ADVERSARIAL GENERATOR — contrary twin of any idea
# ─────────────────────────────────────────────────────────────────────────────

def adversarial_prompt(idea: Dict[str, Any]) -> Dict[str, str]:
    """
    Build a prompt that asks the LLM to generate the contrary version of an idea.
    Returns {system, user} dict ready for an LLM call.
    """
    title = idea.get("title", "")
    method = idea.get("method", "")
    hypothesis = idea.get("hypothesis", "")

    system = (
        "You are a contrarian research scientist. Given a research idea, "
        "identify its CORE ASSUMPTION and propose a research direction that "
        "INVERTS that assumption. The result should be scientifically credible. "
        "Return JSON with the same fields as a normal idea, plus a key "
        "'inverted_assumption' explaining what you flipped."
    )
    user = (
        f"Original idea:\n"
        f"Title: {title}\n"
        f"Hypothesis: {hypothesis}\n"
        f"Method: {method}\n\n"
        f"Steps:\n"
        f"1. Identify the single most load-bearing assumption (e.g., 'larger models are better').\n"
        f"2. Invert it ('what if smaller is better, and how would we prove it?').\n"
        f"3. Generate a complete idea exploring the inverted direction.\n"
        f"Return JSON: {{title, motivation, method, hypothesis, resources, "
        f"expected_outcome, risk_assessment, methodology_type, novelty_level, "
        f"inverted_assumption}}"
    )
    return {"system": system, "user": user}


def fallback_adversarial(idea: Dict[str, Any]) -> Dict[str, Any]:
    """Quick non-LLM adversarial twin (heuristic only)."""
    h = idea.get("hypothesis", "")
    return {
        **idea,
        "title": f"[Contrary] {idea.get('title', '')[:55]}",
        "hypothesis": (
            f"Counter-hypothesis: the assumption underlying '{h[:80]}' may be wrong. "
            f"Explore the opposite direction."
        ),
        "inverted_assumption": "Assumes premise can be inverted",
        "_adversarial": True,
        "_parent_title": idea.get("title", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. FMEA GENERATOR — structured failure mode & effects analysis
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FailureMode:
    mode: str           # what fails
    cause: str          # why it fails
    effect: str         # what's the consequence
    severity: int       # 1-5
    detectability: int  # 1-5 (5 = hard to detect = bad)
    mitigation: str     # how to avoid/handle it

    @property
    def risk_priority(self) -> int:
        """RPN = severity * detectability. Higher = worse."""
        return self.severity * self.detectability


# Heuristic failure modes by methodology type
_HEURISTIC_FMEA = {
    "empirical_study": [
        FailureMode("Hyperparameter overfitting", "Tuning on test set", "Inflated reported accuracy", 4, 4,
                    "Use a separate validation set; report results across 3+ seeds"),
        FailureMode("Data leakage", "Train/test split shares samples", "Method appears better than it is", 5, 5,
                    "Audit splits before any training; use stratified k-fold"),
        FailureMode("Insufficient baselines", "Only weak baselines compared", "Apparent gain is trivial", 4, 2,
                    "Include at least 3 strong baselines published in last 2 years"),
    ],
    "theoretical_analysis": [
        FailureMode("Unrealistic assumptions", "Convex/iid/separable assumed", "Theorems don't apply in practice", 4, 3,
                    "Validate empirically on realistic non-convex scenarios"),
        FailureMode("Vacuous bounds", "Bound is too loose to be useful", "Theory has no practical implication", 3, 4,
                    "Compute bound on real datasets; compare to actual performance"),
    ],
    "system_design": [
        FailureMode("Scalability cliff", "Memory grows quadratically with input", "System fails at production scale", 5, 3,
                    "Profile early at 10x expected input size"),
        FailureMode("Engineering debt", "Prototype-quality code", "Unmaintainable downstream", 3, 3,
                    "Write tests covering 80% of code paths from day 1"),
    ],
    "dataset_creation": [
        FailureMode("Annotator bias", "Single labeler", "Labels reflect one viewpoint", 4, 3,
                    "3+ annotators per item; report inter-annotator agreement (Krippendorff's alpha)"),
        FailureMode("Distribution skew", "Sampling not representative", "Models trained on data fail OOD", 5, 4,
                    "Explicitly stratify by demographic + topic + time"),
    ],
}


def generate_fmea_heuristic(idea: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate FMEA table using heuristics (no LLM call)."""
    mt = idea.get("methodology_type", "empirical_study")
    modes = _HEURISTIC_FMEA.get(mt, _HEURISTIC_FMEA["empirical_study"])
    return [asdict(m) | {"risk_priority": m.risk_priority} for m in modes]


def fmea_prompt(idea: Dict[str, Any]) -> Dict[str, str]:
    """LLM prompt for full FMEA generation (returns 5 specific failure modes)."""
    system = (
        "You are a research methodology expert performing a Failure Mode & "
        "Effects Analysis (FMEA). For the given idea, identify the 5 most likely "
        "failure modes with concrete mitigations.\n\n"
        "Return JSON: {\"failure_modes\": [{\"mode\": str, \"cause\": str, "
        "\"effect\": str, \"severity\": 1-5, \"detectability\": 1-5, "
        "\"mitigation\": str}]}\n\n"
        "Severity: 1=cosmetic, 5=fatal to project.\n"
        "Detectability: 1=immediately obvious, 5=silent failure.\n"
        "Mitigation must be specific and actionable (not 'be careful')."
    )
    user = (
        f"Idea:\nTitle: {idea.get('title', '')}\n"
        f"Method: {idea.get('method', '')}\n"
        f"Hypothesis: {idea.get('hypothesis', '')}\n"
        f"Resources: {idea.get('resources', '')}\n\n"
        f"Generate 5 specific failure modes."
    )
    return {"system": system, "user": user}


def fmea_summary(failure_modes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate stats from an FMEA table."""
    if not failure_modes:
        return {"total": 0, "max_rpn": 0, "high_risk_count": 0, "avg_severity": 0}
    rpns = [fm.get("risk_priority", fm.get("severity", 1) * fm.get("detectability", 1))
            for fm in failure_modes]
    sevs = [fm.get("severity", 0) for fm in failure_modes]
    return {
        "total": len(failure_modes),
        "max_rpn": max(rpns),
        "high_risk_count": sum(1 for r in rpns if r >= 12),
        "avg_severity": round(sum(sevs) / len(sevs), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED ENHANCER — applies all enabled enhancements to an idea
# ─────────────────────────────────────────────────────────────────────────────

def enhance_idea(
    idea: Dict[str, Any],
    knobs: Optional[IdeationKnobs] = None,
    topic: str = "",
) -> Dict[str, Any]:
    """
    Apply all enabled enhancements to an idea, attaching metadata.
    Non-destructive: returns a new dict with extra keys.
    """
    knobs = knobs or IdeationKnobs()
    enhanced = dict(idea)

    # Reproducibility scoring
    if knobs.enable_reproducibility:
        enhanced["_reproducibility"] = reproducibility_score(idea)

    # FMEA (heuristic — fast, no LLM call)
    if knobs.enable_fmea:
        modes = generate_fmea_heuristic(idea)
        enhanced["_fmea"] = {
            "failure_modes": modes,
            "summary": fmea_summary(modes),
        }

    # Domain detection
    enhanced["_domain"] = (
        knobs.domain_persona if knobs.domain_persona != "auto"
        else detect_domain(topic)
    )

    # Adversarial twin (heuristic — fast)
    if knobs.enable_adversarial:
        enhanced["_adversarial_twin"] = fallback_adversarial(idea)

    # Knob context (for transparency/debugging)
    enhanced["_knobs"] = {
        "creativity": knobs.creativity_level,
        "time_weeks": knobs.time_budget_weeks,
        "risk": knobs.risk_tolerance,
        "temperature": knobs.temperature(),
    }
    return enhanced


def build_enhancement_prompt_suffix(knobs: IdeationKnobs, topic: str = "") -> str:
    """
    Build the full prompt suffix to inject into ideation calls.
    Combines knob context + reproducibility + domain persona.
    """
    parts = [knobs.to_prompt_context()]
    if knobs.enable_reproducibility:
        parts.append(REPRODUCIBILITY_PROMPT_INJECT)
    parts.append(domain_persona_prompt(knobs.domain_persona, topic))
    return "\n".join(parts)
