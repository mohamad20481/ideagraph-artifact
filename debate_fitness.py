"""
debate_fitness.py — adversarial-debate-derived fitness (Extension 1 lite).

Background (Extension 1 from the paper roadmap):
    Standard LLM-as-judge fitness is gameable: generators can exploit
    surface features (longer, more jargon-y proposals) to maximize
    the score without producing genuine scientific value (reward
    hacking). The proposed fix is a GRPO-style RL loop where a
    Proposer policy and a Critic policy co-evolve. Training those
    policies is weeks of GPU work — outside this codebase's scope.

    The **lite** version implemented here: keep the *adversarial-game*
    mathematical structure but evaluate it with frozen-weight LLM
    calls instead of trained RL policies. Concretely, for each idea
    H we run T rounds of debate:

        for t in 1..T:
            proposer says <argument defending H>
            critic    says <attack on H or last proposer turn>
            judge     scores both turns on a 0-1 scale

    The idea's fitness is the sigmoid of the score margin:

        F(H) = σ(  Σ_t score(proposer_t) − Σ_t score(critic_t)  )

    Range [0, 1]. F ≈ 0.5 means the debate was a wash; F → 1 means
    H survived the attack (high fitness); F → 0 means the critic
    dismantled H (low fitness). Multiplying the margin by an inverse-
    temperature `τ` (default 2.0) gives the sigmoid more contrast.

    The math is honest to the GRPO formula in the paper — we just
    substitute LLM-as-judge calls for the trained reward model.

Cost: 3T LLM calls per idea (proposer + critic + judge per round).
At T=2 → 6 calls. Disk-cached by hypothesis hash so reruns are free.

Public API:
    DebateFitnessResult                          → dataclass
    compute_debate_fitness(idea, ...)            → DebateFitnessResult
    compute_fitness_for_idea(idea, ...)          → dict   (stamps meta)
    cached_compute_fitness_for_idea(idea, ...)   → dict   (with disk cache)
    debate_fitness_key(idea)                     → float  (sort integration)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


_AUTOLOAD = object()


# ── System prompts ──────────────────────────────────────────────────────────

_PROPOSER_SYS = (
    "You are the PROPOSER. You defend a research hypothesis against an "
    "adversarial critic. Your reply each round is one tight paragraph "
    "(80-150 words): cite a specific reason the hypothesis is plausible "
    "AND directly counter the critic's last attack (if any). Do not "
    "concede; if the critic's point is partially valid, refine and "
    "fight back. Avoid generic enthusiasm — be technically specific."
)

_CRITIC_SYS = (
    "You are the CRITIC. You attack a research hypothesis. Each round "
    "find ONE concrete weakness: an unstated assumption, a missing "
    "baseline, a confounding variable, a known counter-example, or a "
    "step that won't scale. One tight paragraph (80-150 words). Do not "
    "soften the blow; the goal is to find what would kill this idea "
    "in peer review. If the proposer rebuts your last attack, escalate "
    "with a sharper one — do not concede."
)

_JUDGE_SYS = (
    "You are the JUDGE. You score one debate round on a 0-1 scale, "
    "where 0 = the speaker's argument is empty or wrong, 0.5 = mixed, "
    "1 = the argument is rigorous and technically correct. Reply with "
    "STRICT JSON only:\n"
    '{"proposer_score": <0-1>, "critic_score": <0-1>, '
    '"rationale": "<short justification>"}'
)


# ── Math ────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Numerically-stable σ(x) ∈ [0, 1]."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def fitness_from_margin(
    proposer_scores: List[float],
    critic_scores: List[float],
    tau: float = 2.0,
) -> float:
    """Apply the σ(τ · Σ-margin) transform from the GRPO formulation.

    Args:
        proposer_scores: judge's score for each proposer turn (0..1)
        critic_scores:   judge's score for each critic turn   (0..1)
        tau:             inverse-temperature multiplier. Higher = sharper
                         decisions (more polarized fitness). 2.0 is a
                         decent default — F(0.5 margin) ≈ 0.73.

    Returns:
        Fitness ∈ [0, 1]. 0.5 = tied debate; → 1 = proposer dominated.
    """
    margin = sum(proposer_scores or []) - sum(critic_scores or [])
    return _sigmoid(float(tau) * margin)


# ── JSON parsing for judge replies ──────────────────────────────────────────

_NUM_RE = re.compile(r"([0-9]*\.?[0-9]+)")


def _parse_judge_reply(raw: str) -> Dict[str, float]:
    """Extract proposer_score and critic_score from the judge's text.

    Tolerant: strips ``` fences, trims to outer braces, falls back to
    `0.5 / 0.5` on any failure so a flaky judge doesn't crash the loop.
    """
    if not raw:
        return {"proposer_score": 0.5, "critic_score": 0.5}
    s = raw.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl > 0:
            s = s[nl + 1:]
        if s.endswith("```"):
            s = s[:-3].strip()
    a, b = s.find("{"), s.rfind("}")
    if a >= 0 and b > a:
        s = s[a:b + 1]
    try:
        parsed = json.loads(s)
        p = float(parsed.get("proposer_score", 0.5))
        c = float(parsed.get("critic_score", 0.5))
        return {
            "proposer_score": max(0.0, min(1.0, p)),
            "critic_score": max(0.0, min(1.0, c)),
        }
    except Exception:
        pass
    # Fallback: try to pull the two leading floats from the prose.
    nums = _NUM_RE.findall(s)
    if len(nums) >= 2:
        try:
            return {
                "proposer_score": max(0.0, min(1.0, float(nums[0]))),
                "critic_score": max(0.0, min(1.0, float(nums[1]))),
            }
        except Exception:
            pass
    return {"proposer_score": 0.5, "critic_score": 0.5}


# ── LLM helpers ─────────────────────────────────────────────────────────────

def _resolve_client(llm_client: Any) -> Any:
    if llm_client is not _AUTOLOAD:
        return llm_client
    try:
        from claude_provider import get_claude_client
        return get_claude_client()
    except Exception:
        return None


def _safe_call(
    client: Any, system: str, user: str,
    max_tokens: int = 250, temperature: float = 0.7,
) -> str:
    if client is None:
        return ""
    try:
        resp = client.call(
            system=system, user=user,
            max_tokens=max_tokens, temperature=temperature,
        )
    except Exception:
        return ""
    if not getattr(resp, "success", False):
        return ""
    return (getattr(resp, "text", "") or "").strip()


def _hypothesis_block(idea: Dict[str, Any]) -> str:
    """Render the idea as a "Hypothesis under debate" block for context."""
    fields = [
        ("Title",       idea.get("title")),
        ("Hypothesis",  idea.get("hypothesis")),
        ("Motivation",  idea.get("motivation")),
        ("Method",      idea.get("method")),
        ("Risks",       idea.get("risk_assessment")),
    ]
    return "\n".join(
        f"  {k}: {v}" for k, v in fields if v
    )


# ── Round runner ────────────────────────────────────────────────────────────

@dataclass
class DebateRound:
    proposer_text: str
    critic_text: str
    proposer_score: float
    critic_score: float
    judge_rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DebateFitnessResult:
    """Outcome of running T debate rounds for one idea."""
    fitness: float                        # σ(τ · margin) ∈ [0, 1]
    proposer_total: float                 # Σ proposer_scores
    critic_total: float                   # Σ critic_scores
    margin: float                         # proposer_total − critic_total
    n_rounds: int
    tau: float
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    cache_key: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_debate_fitness(
    idea: Dict[str, Any],
    llm_client: Any = _AUTOLOAD,
    n_rounds: int = 2,
    tau: float = 2.0,
    max_tokens: int = 250,
) -> DebateFitnessResult:
    """Run T rounds of Proposer/Critic debate and compute σ(τ·margin).

    Returns a zero-margin (fitness=0.5) result if the LLM is unavailable,
    so the loop never crashes — fitness collapses to neutral, the rest
    of the pipeline keeps moving.
    """
    cache_key = _hash_key(idea, n_rounds, tau)
    if not isinstance(idea, dict):
        return DebateFitnessResult(
            fitness=0.5, proposer_total=0.0, critic_total=0.0,
            margin=0.0, n_rounds=0, tau=float(tau), cache_key=cache_key,
        )
    client = _resolve_client(llm_client)
    if client is None:
        return DebateFitnessResult(
            fitness=0.5, proposer_total=0.0, critic_total=0.0,
            margin=0.0, n_rounds=0, tau=float(tau), cache_key=cache_key,
        )

    hyp_block = _hypothesis_block(idea)
    transcript: List[str] = []         # rolling context shown to all agents
    rounds: List[DebateRound] = []

    for t in range(max(1, int(n_rounds))):
        last_critic = (
            transcript[-1] if transcript and transcript[-1].startswith("CRITIC")
            else ""
        )
        proposer_user = (
            f"=== HYPOTHESIS ===\n{hyp_block}\n\n"
            + (f"=== CRITIC'S LAST POINT ===\n{last_critic}\n\n"
               if last_critic else "")
            + "Defend the hypothesis. One tight paragraph (80-150 words)."
        )
        proposer_text = _safe_call(
            client, _PROPOSER_SYS, proposer_user,
            max_tokens=max_tokens, temperature=0.7,
        )
        transcript.append(f"PROPOSER (round {t+1}): {proposer_text}")

        critic_user = (
            f"=== HYPOTHESIS ===\n{hyp_block}\n\n"
            f"=== PROPOSER'S LAST DEFENSE ===\n{proposer_text}\n\n"
            "Attack the hypothesis with one concrete weakness. "
            "One tight paragraph (80-150 words)."
        )
        critic_text = _safe_call(
            client, _CRITIC_SYS, critic_user,
            max_tokens=max_tokens, temperature=0.8,
        )
        transcript.append(f"CRITIC (round {t+1}): {critic_text}")

        # Judge scores BOTH turns of this round at once.
        judge_user = (
            f"=== HYPOTHESIS ===\n{hyp_block}\n\n"
            f"=== ROUND {t+1} ===\n"
            f"Proposer said: {proposer_text}\n\n"
            f"Critic said:   {critic_text}\n\n"
            "Score both speakers (0-1 each). JSON only."
        )
        judge_raw = _safe_call(
            client, _JUDGE_SYS, judge_user,
            max_tokens=120, temperature=0.0,
        )
        scores = _parse_judge_reply(judge_raw)
        rounds.append(DebateRound(
            proposer_text=proposer_text,
            critic_text=critic_text,
            proposer_score=scores["proposer_score"],
            critic_score=scores["critic_score"],
            judge_rationale="",
        ))

    p_scores = [r.proposer_score for r in rounds]
    c_scores = [r.critic_score for r in rounds]
    fitness = fitness_from_margin(p_scores, c_scores, tau=tau)
    p_total = sum(p_scores)
    c_total = sum(c_scores)
    return DebateFitnessResult(
        fitness=fitness,
        proposer_total=p_total,
        critic_total=c_total,
        margin=p_total - c_total,
        n_rounds=len(rounds),
        tau=float(tau),
        rounds=[r.to_dict() for r in rounds],
        cache_key=cache_key,
    )


# ── Idea integration ──────────────────────────────────────────────────────

def compute_fitness_for_idea(
    idea: Dict[str, Any],
    llm_client: Any = _AUTOLOAD,
    n_rounds: int = 2,
    tau: float = 2.0,
    persist_to_meta: bool = True,
) -> Dict[str, Any]:
    """Run debate fitness on an idea and (optionally) stamp the result
    onto `idea["execution_meta"]["debate_fitness"]`."""
    result = compute_debate_fitness(
        idea, llm_client=llm_client,
        n_rounds=n_rounds, tau=tau,
    )
    payload = result.to_dict()
    if persist_to_meta and isinstance(idea, dict):
        meta = idea.get("execution_meta")
        if not isinstance(meta, dict):
            meta = {}
        meta["debate_fitness"] = payload
        idea["execution_meta"] = meta
    return payload


def debate_fitness_key(idea: Dict[str, Any]) -> float:
    """Read cached debate fitness from execution_meta. Returns 0.5
    (neutral) if not yet computed — so unscored ideas don't bias the
    sort all the way to 0 or 1."""
    if not isinstance(idea, dict):
        return 0.5
    meta = idea.get("execution_meta") or {}
    df = meta.get("debate_fitness") or {}
    try:
        v = float(df.get("fitness", 0.5) or 0.5)
    except (TypeError, ValueError):
        return 0.5
    if not df:
        return 0.5
    return v


def debate_margin_key(idea: Dict[str, Any]) -> float:
    """Read raw signed margin (Σ proposer − Σ critic). Used by the
    'debate_margin' sort mode which ignores σ-compression."""
    if not isinstance(idea, dict):
        return 0.0
    meta = idea.get("execution_meta") or {}
    df = meta.get("debate_fitness") or {}
    try:
        return float(df.get("margin", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


# ── Caching ────────────────────────────────────────────────────────────────

def _hash_key(idea: Any, n_rounds: int, tau: float) -> str:
    """Hash of idea identity + run parameters. Tolerates non-dict input
    by hashing the repr — so the caller always gets a stable cache_key
    even when graceful-degradation returns a neutral result."""
    if not isinstance(idea, dict):
        base = repr(idea) + f"||T={int(n_rounds)}||tau={float(tau):.3f}"
    else:
        base = (
            (idea.get("title") or "")
            + "||"
            + (idea.get("hypothesis") or "")
            + "||"
            + (idea.get("motivation") or "")
            + f"||T={int(n_rounds)}||tau={float(tau):.3f}"
        )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


_DEFAULT_CACHE_DIR = ".ideagraph_debate_cache"


def cached_compute_fitness_for_idea(
    idea: Dict[str, Any],
    llm_client: Any = _AUTOLOAD,
    n_rounds: int = 2,
    tau: float = 2.0,
    cache_dir: str = _DEFAULT_CACHE_DIR,
) -> Dict[str, Any]:
    """Same as compute_fitness_for_idea but persists each result to a
    JSON file keyed by SHA-256 of (idea identity, n_rounds, tau)."""
    key = _hash_key(idea, n_rounds, tau)
    path = os.path.join(cache_dir, f"{key}.json")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            meta = idea.get("execution_meta") or {}
            meta["debate_fitness"] = payload
            idea["execution_meta"] = meta
            return payload
    except Exception:
        pass

    payload = compute_fitness_for_idea(
        idea, llm_client=llm_client,
        n_rounds=n_rounds, tau=tau, persist_to_meta=True,
    )
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return payload
