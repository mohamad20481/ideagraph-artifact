"""
idea_regenerator.py — produce derivative ideas from an existing parent idea.

Given an idea the user already has (from the current run, history, or a
shared link), regenerate N variants using one of six modes:

  🔬 Refine     — same direction, address probe weaknesses
  🌳 Extend     — take this as a stepping stone, go bigger
  ↔️ Pivot       — same problem, different methodology
  🎭 Contrast   — adversarial counter-idea (what a skeptic would propose)
  🌐 Cross-domain — apply the structure to a totally different field
  🎲 Mutate     — small perturbations (dataset, method, scale)

Each new idea inherits `parent_title` and gets `generation = parent.generation + 1`
so the lineage is preserved in the existing Idea dataclass.

Public API:
    REGEN_MODES                                          → Dict[mode_id, dict]
    regenerate(parent, mode, n=1, claude_client=None)    → List[Idea]
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, fields
from typing import Any, Dict, List, Optional

from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


# ─────────────────────────────────────────────────────────────────────────────
# Mode catalog — drives the regeneration prompt and the UI labels
# ─────────────────────────────────────────────────────────────────────────────

REGEN_MODES: Dict[str, Dict[str, Any]] = {
    "refine": {
        "label": "🔬 Refine",
        "tagline": "Same direction, fixed weaknesses",
        "description": (
            "Same research direction as the parent, but tighter: address "
            "specific probe weaknesses, swap vague phrasing for concrete "
            "specifications, add the missing technical details."
        ),
        "instruction": (
            "Refine the parent idea: keep its core direction and methodology, "
            "but strengthen weak components. Address the specific concerns "
            "below. Make the method more concrete (named algorithms, exact "
            "datasets, hyperparameter ranges). Make the hypothesis falsifiable "
            "with quantitative success criteria. Make resource estimates "
            "specific (GPU-hours, dataset size, wall-clock time)."
        ),
        "preserve_methodology": True,
        "preserve_novelty": True,
        "default_temp": 0.55,
    },
    "extend": {
        "label": "🌳 Extend",
        "tagline": "Build on this as a stepping stone",
        "description": (
            "Treat the parent as Phase 1. Propose Phase 2: bigger scope, "
            "broader population, longer timeline, deeper analysis. The new "
            "idea takes the parent's success as given and goes further."
        ),
        "instruction": (
            "Treat the parent idea as a successful Phase 1. Propose a "
            "follow-on idea that ASSUMES the parent's hypothesis was "
            "confirmed. The new idea should be MORE AMBITIOUS in at least "
            "one dimension: broader population, multi-modal extension, "
            "longer timeline, or qualitatively new capability. Cite the "
            "parent explicitly in the motivation."
        ),
        "preserve_methodology": False,
        "preserve_novelty": False,
        "default_temp": 0.70,
    },
    "pivot": {
        "label": "↔️ Pivot",
        "tagline": "Same problem, different methodology",
        "description": (
            "Keep the parent's problem and target outcome but swap the "
            "methodology entirely (e.g., empirical → theoretical, system "
            "design → dataset creation). Forces the same goal through a "
            "different lens."
        ),
        "instruction": (
            "Pivot the parent idea: keep the SAME research question, target "
            "outcome, and motivation, but propose a DIFFERENT methodology "
            "type. If parent is empirical, propose theoretical or system "
            "design. If parent is theoretical, propose empirical or dataset. "
            "The new methodology must genuinely address the parent's "
            "hypothesis from a fresh angle, not just rebrand it."
        ),
        "preserve_methodology": False,
        "preserve_novelty": True,
        "default_temp": 0.75,
    },
    "contrast": {
        "label": "🎭 Contrast",
        "tagline": "What a skeptic would propose",
        "description": (
            "Adversarial counter-idea. Propose a research idea that would "
            "be the natural rebuttal to the parent — testing the same "
            "claim from the opposite premise. Useful for stress-testing."
        ),
        "instruction": (
            "Generate an ADVERSARIAL counter-idea to the parent. Imagine a "
            "skeptic who suspects the parent's hypothesis is wrong. What "
            "would they propose to falsify it? The new idea should: "
            "(1) target the parent's main claim directly, "
            "(2) use a methodology that would expose the parent's blind "
            "spots, (3) state a hypothesis that contradicts or strongly "
            "qualifies the parent's. Be respectful but rigorous."
        ),
        "preserve_methodology": False,
        "preserve_novelty": False,
        "default_temp": 0.80,
    },
    "cross_domain": {
        "label": "🌐 Cross-domain",
        "tagline": "Same structure, different field",
        "description": (
            "Take the parent's STRUCTURAL idea (the technical insight) and "
            "apply it in a completely different research domain. Materials "
            "→ biology, NLP → robotics, physics → economics."
        ),
        "instruction": (
            "Identify the structural / methodological insight at the heart "
            "of the parent idea (the technique, not the application). Now "
            "transplant it into a completely different research domain — "
            "ideally one with similar structural problems but different "
            "vocabulary. Explain the analogy explicitly. The new idea "
            "should read as a credible proposal in the target domain, not "
            "a thin re-skin."
        ),
        "preserve_methodology": False,
        "preserve_novelty": False,
        "default_temp": 0.85,
    },
    "mutate": {
        "label": "🎲 Mutate",
        "tagline": "Small perturbations",
        "description": (
            "Keep most of the parent intact and perturb one or two "
            "components: swap the dataset, replace one architectural choice, "
            "change the scale, alter one assumption. Useful when you mostly "
            "like the parent."
        ),
        "instruction": (
            "Mutate the parent idea by changing exactly ONE or TWO "
            "components. Keep the title, motivation, and overall direction "
            "very close to the parent. Possible mutations: (a) swap the "
            "dataset for a related but distinct one, (b) replace one "
            "architectural component with an alternative, (c) shift the "
            "scale up or down by an order of magnitude, (d) change one "
            "key assumption. State explicitly which components changed and "
            "why."
        ),
        "preserve_methodology": True,
        "preserve_novelty": True,
        "default_temp": 0.65,
    },
    "topic_transplant": {
        "label": "🎯 Topic Transplant",
        "tagline": "Same idea, different topic",
        "description": (
            "Keep the parent's methodology, technical structure, and "
            "experimental design as much as possible — but apply it to a "
            "completely different topic that you specify. Use this when you "
            "want the same recipe in a new domain (e.g., 'this GNN approach, "
            "but for materials discovery instead of drug discovery')."
        ),
        "instruction": (
            "Transplant the parent idea to a NEW topic that the user is "
            "providing. Preserve the parent's methodology, technical "
            "approach, evaluation strategy, and overall structure as "
            "faithfully as possible — only the *application domain* should "
            "change. Update the title, motivation, hypothesis, expected "
            "outcome, and resources to match the new topic. Be explicit "
            "about which structural elements you kept and which you adapted."
        ),
        "preserve_methodology": True,
        "preserve_novelty": True,
        "default_temp": 0.60,
        "requires_target_topic": True,
    },
}


_SYSTEM = (
    "You are an expert research scientist. The user gives you an existing "
    "research idea (the 'parent') and a regeneration mode. Your job is to "
    "produce ONE new research idea derived from the parent according to the "
    "mode's instructions. Output ONLY valid JSON with exactly these keys: "
    "title, motivation, method, hypothesis, resources, expected_outcome, "
    "risk_assessment, source_strategy, methodology_type, novelty_level, "
    "lineage_note. "
    f"methodology_type must be one of: {', '.join(METHODOLOGY_TYPES)}. "
    f"novelty_level must be one of: {', '.join(NOVELTY_LEVELS)}. "
    "lineage_note is a single sentence explaining how this derives from the "
    "parent."
)


def _parent_summary(parent: Any) -> Dict[str, str]:
    """Pull the seven core fields out of an Idea or dict."""
    if hasattr(parent, "to_dict"):
        d = parent.to_dict()
    elif isinstance(parent, dict):
        d = parent
    else:
        return {}
    return {
        k: str(d.get(k, ""))
        for k in (
            "title", "motivation", "method", "hypothesis",
            "resources", "expected_outcome", "risk_assessment",
            "methodology_type", "novelty_level",
        )
    }


def _build_user_prompt(parent: Any, mode: str,
                        weak_probes: Optional[Dict[str, float]] = None,
                        target_topic: str = "") -> str:
    cfg = REGEN_MODES[mode]
    p = _parent_summary(parent)
    weak_section = ""
    if weak_probes:
        weak_lines = [
            f"  - {k}: {v:.2f}" for k, v in sorted(weak_probes.items(), key=lambda kv: kv[1])
            if isinstance(v, (int, float)) and v < 0.6
        ][:5]
        if weak_lines:
            weak_section = (
                "\nProbe weaknesses to address (lower = worse):\n"
                + "\n".join(weak_lines) + "\n"
            )
    constraint_lines = []
    if cfg.get("preserve_methodology") and p.get("methodology_type"):
        constraint_lines.append(
            f"Constraint: methodology_type MUST stay '{p['methodology_type']}'."
        )
    if cfg.get("preserve_novelty") and p.get("novelty_level"):
        constraint_lines.append(
            f"Constraint: novelty_level should stay around '{p['novelty_level']}' "
            "(may shift one step up or down)."
        )
    constraints = ("\n" + "\n".join(constraint_lines) + "\n") if constraint_lines else ""

    # Target-topic injection — applies to every mode when set; required for
    # topic_transplant. The instruction is intentionally explicit so the LLM
    # doesn't accidentally drift back to the parent's domain.
    target_topic = (target_topic or "").strip()
    target_section = ""
    if target_topic:
        target_section = (
            f"\n### NEW target topic (override)\n"
            f"Apply the regeneration to this domain instead of the parent's:\n"
            f"  >>> {target_topic} <<<\n"
            f"All five generated fields (title, motivation, method, hypothesis, "
            f"expected_outcome) MUST be specifically about this new topic. Do "
            f"not silently revert to the parent's domain. If the parent's "
            f"methodology genuinely doesn't transfer, say so in lineage_note "
            f"and adapt as needed.\n"
        )

    return (
        f"## Mode: {cfg['label']} — {cfg['tagline']}\n\n"
        f"### Parent idea\n"
        f"  Title: {p.get('title','')}\n"
        f"  Motivation: {p.get('motivation','')[:300]}\n"
        f"  Method: {p.get('method','')[:400]}\n"
        f"  Hypothesis: {p.get('hypothesis','')[:300]}\n"
        f"  Resources: {p.get('resources','')[:200]}\n"
        f"  Expected outcome: {p.get('expected_outcome','')[:200]}\n"
        f"  Risk assessment: {p.get('risk_assessment','')[:200]}\n"
        f"  methodology_type: {p.get('methodology_type','')}\n"
        f"  novelty_level: {p.get('novelty_level','')}\n"
        f"{weak_section}"
        f"{target_section}"
        f"\n### Instructions\n{cfg['instruction']}\n"
        f"{constraints}"
        "\n### Output\n"
        "Return JSON:\n"
        "{\n"
        '  "title": "<concise title for the NEW idea>",\n'
        '  "motivation": "<why this matters; reference the parent if relevant>",\n'
        '  "method": "<concrete technical approach>",\n'
        '  "hypothesis": "<testable prediction>",\n'
        '  "resources": "<datasets, compute, software needed>",\n'
        '  "expected_outcome": "<measurable results>",\n'
        '  "risk_assessment": "<main risks and mitigations>",\n'
        '  "source_strategy": "R",\n'
        f'  "methodology_type": "<one of: {", ".join(METHODOLOGY_TYPES)}>",\n'
        f'  "novelty_level": "<one of: {", ".join(NOVELTY_LEVELS)}>",\n'
        '  "lineage_note": "<one sentence explaining how this derives from parent>"\n'
        "}"
    )


def _parse_idea_json(raw: str) -> Optional[Dict[str, Any]]:
    """Parse LLM response, tolerating fenced code blocks."""
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        s = s[start:end + 1]
    try:
        return json.loads(s)
    except Exception:
        return None


def _dict_to_idea(d: Dict[str, Any], parent: Any) -> Optional[Idea]:
    """Build an Idea dataclass from the parsed LLM response."""
    if not d:
        return None
    required = ("title", "method", "hypothesis")
    for k in required:
        if not str(d.get(k, "")).strip():
            return None

    method_type = d.get("methodology_type") or ""
    if method_type not in METHODOLOGY_TYPES:
        method_type = None
    novelty = d.get("novelty_level") or ""
    if novelty not in NOVELTY_LEVELS:
        novelty = None

    parent_dict = (parent.to_dict() if hasattr(parent, "to_dict")
                    else (parent if isinstance(parent, dict) else {}))
    parent_gen = int(parent_dict.get("generation") or 0)

    idea = Idea(
        title=str(d.get("title", ""))[:200],
        motivation=str(d.get("motivation", ""))[:1000],
        method=str(d.get("method", ""))[:2000],
        hypothesis=str(d.get("hypothesis", ""))[:1000],
        resources=str(d.get("resources", ""))[:500],
        expected_outcome=str(d.get("expected_outcome", ""))[:500],
        risk_assessment=str(d.get("risk_assessment", ""))[:500],
        source_strategy="R",  # "R" for Regeneration
        methodology_type=method_type,
        novelty_level=novelty,
        generation=parent_gen + 1,
        parent_title=str(parent_dict.get("title", ""))[:200],
    )
    # Stash the lineage note + mode on execution_meta for the UI to surface
    idea.execution_meta = {
        "lineage_note": str(d.get("lineage_note", ""))[:400],
        "regen_mode": d.get("_regen_mode", ""),
    }
    return idea


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel and main entry point
# ─────────────────────────────────────────────────────────────────────────────

_AUTOLOAD = object()


# ─────────────────────────────────────────────────────────────────────────────
# Provider-error classifier — turn opaque API errors into actionable hints
# ─────────────────────────────────────────────────────────────────────────────
#
# Most LLM provider errors fall into a handful of well-known buckets that have
# very different remedies. The raw error string from the API is often a JSON
# blob like `API 403: {"code":"INSUFFICIENT_BALANCE",...}` which is opaque
# unless you've seen it before. This classifier matches common patterns and
# returns a one-line hint pointing at the fix.
#
# Returns an empty string for unrecognized errors (so the diagnostic stays
# unchanged for novel error shapes).

def _classify_api_error(err: str) -> str:
    """Match the error string against known patterns; return an
    actionable hint or '' if nothing matches."""
    if not err:
        return ""
    e = err.lower()

    # ── Balance / quota / credit exhausted ─────────────────────────────────
    if (
        "insufficient_balance" in e
        or "insufficient balance" in e
        or "out of credit" in e
        or "quota exceeded" in e
        or "billing" in e and ("paid" in e or "limit" in e)
    ):
        return (
            "💸 Your provider account is out of credit. Top it up, "
            "or switch to a different provider in Admin → 🔌 LLM Provider "
            "(Gemini and Groq have generous free tiers)."
        )

    # ── Authentication / API key issues ────────────────────────────────────
    if (
        "401" in e
        or "invalid_api_key" in e
        or "invalid api key" in e
        or "incorrect api key" in e
        or "authentication" in e
        or "unauthorized" in e
    ):
        return (
            "🔑 The API key is missing, wrong, or revoked. Check the "
            "active provider's key in Admin → 🔌 LLM Provider, or in your "
            ".env file."
        )

    # ── Rate-limit / throttling ────────────────────────────────────────────
    if (
        "rate_limit" in e
        or "rate limit" in e
        or "429" in e
        or "too many requests" in e
        or "throttle" in e
    ):
        return (
            "⏳ Rate-limited by the provider. Wait ~30s and retry, "
            "lower `n` (variants per click), or switch to a less-busy "
            "provider in Admin → 🔌 LLM Provider."
        )

    # ── Upstream / proxy errors (aiprimetech proxy is known-flaky) ────────
    if (
        "502" in e or "503" in e or "504" in e
        or "bad gateway" in e or "gateway timeout" in e
        or "service unavailable" in e
    ):
        return (
            "🌐 Upstream provider returned a transient error. Retry in "
            "~10s. If it persists, switch providers (Admin → 🔌 LLM "
            "Provider) — the aiprimetech.io proxy is known to be flaky."
        )

    # ── Context length exceeded ────────────────────────────────────────────
    if (
        "context_length" in e
        or "maximum context length" in e
        or "context window" in e
        or "too long" in e and "context" in e
    ):
        return (
            "📏 The parent idea is too long for this model's context "
            "window. Switch to a model with a larger context (Claude "
            "Sonnet, Gemini Pro), or shorten the parent's method/motivation."
        )

    # ── Model not found / wrong provider+model combination ────────────────
    if (
        "model_not_found" in e
        or "model not found" in e
        or "does not exist" in e and "model" in e
        or "no such model" in e
    ):
        return (
            "🎯 The configured model name isn't recognized by this "
            "provider. Check Admin → 🔌 LLM Provider for valid model "
            "names for your provider."
        )

    return ""


def regenerate(
    parent: Any,
    mode: str,
    n: int = 1,
    claude_client: Any = _AUTOLOAD,
    weak_probes: Optional[Dict[str, float]] = None,
    max_tokens: int = 800,
    target_topic: str = "",
    diagnostics: Optional[List[str]] = None,
) -> List[Idea]:
    """Generate `n` derivative ideas from the parent in the given mode.

    Returns a list (possibly empty if the LLM is unavailable or all calls
    failed to parse). The returned Idea instances have `parent_title`,
    `generation`, `source_strategy='R'`, and `execution_meta['lineage_note']`
    populated.

    `claude_client` semantics match agents/execution_revisor.py: pass an
    object to use it, ``None`` to skip the LLM (returns empty list), or
    omit to auto-load the global client.

    `target_topic` (optional for most modes; required for the
    `topic_transplant` mode) overrides the parent's domain — the LLM is
    instructed to apply the regeneration to the new topic instead.

    `diagnostics` (optional): pass an empty list to collect per-call
    failure reasons (one human-readable string per failed call). The UI
    can surface these in the warning banner so the user knows *why* zero
    variants came back instead of just "no variants returned".
    """
    if mode not in REGEN_MODES:
        raise ValueError(
            f"Unknown regeneration mode '{mode}'. "
            f"Valid: {sorted(REGEN_MODES.keys())}"
        )
    if n <= 0:
        return []

    target_topic = (target_topic or "").strip()
    if REGEN_MODES[mode].get("requires_target_topic") and not target_topic:
        raise ValueError(
            f"Mode '{mode}' requires a non-empty target_topic argument "
            "(the new domain to transplant the idea into)."
        )

    if claude_client is _AUTOLOAD:
        try:
            from claude_provider import get_claude_client
            claude_client = get_claude_client()
        except Exception as _e:
            if diagnostics is not None:
                diagnostics.append(
                    f"LLM client auto-load failed: "
                    f"{type(_e).__name__}: {str(_e)[:120]}"
                )
            claude_client = None

    if claude_client is None:
        if diagnostics is not None:
            diagnostics.append(
                "No LLM client configured. Open Admin Dashboard → "
                "🔌 LLM Provider and verify the provider + API key."
            )
        return []

    cfg = REGEN_MODES[mode]
    out: List[Idea] = []
    user_prompt = _build_user_prompt(
        parent, mode, weak_probes=weak_probes, target_topic=target_topic,
    )

    # Each regeneration is its own LLM call so the temperature jitter
    # produces genuinely different children. We bump temperature slightly
    # per attempt so we don't get N near-duplicates back.
    base_temp = float(cfg.get("default_temp", 0.7))
    for k in range(n):
        temp = min(0.95, base_temp + 0.05 * k)
        try:
            resp = claude_client.call(
                system=_SYSTEM,
                user=user_prompt,
                max_tokens=max_tokens,
                temperature=temp,
                json_mode=True,
            )
        except Exception as _e:
            if diagnostics is not None:
                diagnostics.append(
                    f"Call {k+1}/{n}: client exception "
                    f"({type(_e).__name__}): {str(_e)[:160]}"
                )
            continue
        if not getattr(resp, "success", False):
            if diagnostics is not None:
                err = getattr(resp, "error", "") or "unknown error"
                err_str = str(err)[:300]
                hint = _classify_api_error(err_str)
                msg = (
                    f"Call {k+1}/{n}: API returned failure — "
                    f"{err_str}"
                )
                if hint:
                    msg += f"\n    → {hint}"
                diagnostics.append(msg)
            continue
        parsed = _parse_idea_json(getattr(resp, "text", ""))
        if not parsed:
            if diagnostics is not None:
                txt = getattr(resp, "text", "") or ""
                diagnostics.append(
                    f"Call {k+1}/{n}: response was not valid idea JSON "
                    f"({len(txt)} chars received). Try a different model "
                    f"(some smaller models struggle with strict JSON)."
                )
            continue
        parsed["_regen_mode"] = mode
        idea = _dict_to_idea(parsed, parent)
        if idea is not None:
            # Attach target_topic to the lineage record so the UI can show
            # what domain the idea was transplanted to.
            if target_topic and idea.execution_meta is not None:
                idea.execution_meta["target_topic"] = target_topic
            out.append(idea)
        elif diagnostics is not None:
            diagnostics.append(
                f"Call {k+1}/{n}: JSON parsed but lacked required "
                f"fields (title / method / hypothesis)."
            )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Fresh-take generation: same topic, different sources/angles
# ─────────────────────────────────────────────────────────────────────────────
#
# Different shape of regeneration: instead of deriving from a single parent,
# we generate fresh ideas on the SAME topic that explicitly avoid duplicating
# any of the existing ideas. The existing ideas are passed as anti-exemplars,
# and the LLM is instructed to use a different methodology, different
# theoretical framing, and different intellectual lineage.

_FRESH_SYSTEM = (
    "You are an expert research scientist generating ideas on a given topic. "
    "The user provides a topic and a list of ideas that have ALREADY been "
    "generated on that topic — your job is to produce ONE genuinely "
    "different new idea, grounded in different sources, methodology, and "
    "framing than any of the existing ideas. Output ONLY valid JSON with "
    "exactly these keys: title, motivation, method, hypothesis, resources, "
    "expected_outcome, risk_assessment, source_strategy, methodology_type, "
    "novelty_level, divergence_note. "
    f"methodology_type must be one of: {', '.join(METHODOLOGY_TYPES)}. "
    f"novelty_level must be one of: {', '.join(NOVELTY_LEVELS)}. "
    "divergence_note is a single sentence explaining what makes this new "
    "idea source/angle distinct from the existing ones."
)


def _summarize_existing_ideas(ideas: List[Any], max_chars: int = 200) -> str:
    """Compact bullet list of existing ideas to use as anti-exemplars."""
    lines = []
    for i, it in enumerate(ideas[:12], 1):
        d = it.to_dict() if hasattr(it, "to_dict") else (it if isinstance(it, dict) else {})
        title = str(d.get("title", "(untitled)"))
        method_type = d.get("methodology_type") or "?"
        novelty = d.get("novelty_level") or "?"
        method_summary = str(d.get("method", ""))[:max_chars]
        lines.append(
            f"  {i}. [{method_type} × {novelty}] {title}\n"
            f"     method: {method_summary}…"
        )
    return "\n".join(lines)


def _existing_methodology_distribution(ideas: List[Any]) -> Dict[str, int]:
    """Count how many existing ideas use each methodology_type."""
    counts: Dict[str, int] = {}
    for it in ideas:
        d = it.to_dict() if hasattr(it, "to_dict") else (it if isinstance(it, dict) else {})
        m = d.get("methodology_type") or "?"
        counts[m] = counts.get(m, 0) + 1
    return counts


def _build_fresh_user_prompt(
    topic: str,
    existing_ideas: List[Any],
    avoid_methodologies: bool = True,
) -> str:
    """Prompt that lists existing ideas as anti-exemplars and demands a
    genuinely different angle."""
    existing_summary = _summarize_existing_ideas(existing_ideas)
    method_dist = _existing_methodology_distribution(existing_ideas)
    underrep_methodologies = [
        m for m in METHODOLOGY_TYPES if method_dist.get(m, 0) == 0
    ]

    avoid_section = ""
    if avoid_methodologies and underrep_methodologies:
        avoid_section = (
            "\n### Methodology diversity\n"
            "These methodology_types haven't been used by existing ideas — "
            "STRONGLY prefer one of them for genuine diversity:\n"
            + "\n".join(f"  - {m}" for m in underrep_methodologies[:6])
            + "\n"
            + "Used methodology_types (over-represented): "
            + ", ".join(f"{m}({c})" for m, c in method_dist.items()
                         if m != "?")
            + "\n"
        )

    return (
        f"## Mode: 🔀 Fresh Take — same topic, different sources\n\n"
        f"### TOPIC\n  {topic}\n\n"
        f"### Existing ideas to AVOID duplicating ({len(existing_ideas)} total)\n"
        f"{existing_summary}\n"
        f"{avoid_section}"
        f"\n### Instructions\n"
        f"Generate ONE new research idea on the SAME topic above, but it MUST "
        f"come from a fundamentally different source/angle than every "
        f"existing idea. Specifically:\n"
        f"  • Pick a methodology_type that is under-represented (preferably "
        f"unused) in the existing list.\n"
        f"  • Frame the problem from a different theoretical, empirical, or "
        f"engineering perspective.\n"
        f"  • Cite different prior work / paradigms as inspiration "
        f"(name them in motivation if relevant).\n"
        f"  • Take a different stance on the central question.\n"
        f"  • Avoid keyword overlap in the title with existing ideas.\n"
        f"\nIn the divergence_note, explicitly name what's different about "
        f"YOUR idea's source compared to the existing set.\n"
        f"\n### Output\n"
        "Return JSON:\n"
        "{\n"
        '  "title": "<concise title>",\n'
        '  "motivation": "<why this matters>",\n'
        '  "method": "<concrete technical approach>",\n'
        '  "hypothesis": "<testable prediction>",\n'
        '  "resources": "<datasets, compute, software needed>",\n'
        '  "expected_outcome": "<measurable results>",\n'
        '  "risk_assessment": "<main risks and mitigations>",\n'
        '  "source_strategy": "F",\n'
        f'  "methodology_type": "<one of: {", ".join(METHODOLOGY_TYPES)}>",\n'
        f'  "novelty_level": "<one of: {", ".join(NOVELTY_LEVELS)}>",\n'
        '  "divergence_note": "<one sentence: how is your source/angle '
        'different from the existing ideas>"\n'
        "}"
    )


def _fresh_dict_to_idea(d: Dict[str, Any], topic: str) -> Optional[Idea]:
    """Build an Idea from the fresh-take LLM response. No parent — these
    are top-level new ideas, generation 0, source_strategy 'F'."""
    if not d:
        return None
    if not all(str(d.get(k, "")).strip() for k in ("title", "method", "hypothesis")):
        return None

    method_type = d.get("methodology_type") or ""
    if method_type not in METHODOLOGY_TYPES:
        method_type = None
    novelty = d.get("novelty_level") or ""
    if novelty not in NOVELTY_LEVELS:
        novelty = None

    idea = Idea(
        title=str(d.get("title", ""))[:200],
        motivation=str(d.get("motivation", ""))[:1000],
        method=str(d.get("method", ""))[:2000],
        hypothesis=str(d.get("hypothesis", ""))[:1000],
        resources=str(d.get("resources", ""))[:500],
        expected_outcome=str(d.get("expected_outcome", ""))[:500],
        risk_assessment=str(d.get("risk_assessment", ""))[:500],
        source_strategy="F",  # "F" for Fresh-take
        methodology_type=method_type,
        novelty_level=novelty,
        generation=0,
        parent_title=None,
    )
    idea.execution_meta = {
        "divergence_note": str(d.get("divergence_note", ""))[:400],
        "regen_mode": "fresh_take",
        "topic": topic,
    }
    return idea


def regenerate_fresh(
    topic: str,
    existing_ideas: List[Any],
    n: int = 3,
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 800,
    avoid_methodologies: bool = True,
) -> List[Idea]:
    """Generate `n` new ideas on `topic` that explicitly avoid duplicating
    any of the `existing_ideas`. Uses different methodology distributions
    and a higher base temperature than the parent-derived modes.

    Returns Idea instances with `source_strategy='F'`, `generation=0`, and
    `execution_meta['divergence_note']` populated.
    """
    if not topic or not topic.strip():
        raise ValueError("regenerate_fresh requires a non-empty topic")
    if n <= 0:
        return []

    if claude_client is _AUTOLOAD:
        try:
            from claude_provider import get_claude_client
            claude_client = get_claude_client()
        except Exception:
            claude_client = None

    if claude_client is None:
        return []

    user_prompt = _build_fresh_user_prompt(
        topic, existing_ideas, avoid_methodologies=avoid_methodologies,
    )

    out: List[Idea] = []
    base_temp = 0.85  # higher than parent-mode default — diversity is the goal
    seen_titles: List[str] = []
    for k in range(n):
        # Bump temperature each attempt + tell the LLM about the fresh
        # ideas we've already produced this session, so it doesn't return
        # near-duplicates of THOSE either.
        prompt = user_prompt
        if seen_titles:
            prompt += (
                "\n### Already produced THIS session (avoid these too)\n"
                + "\n".join(f"  - {t}" for t in seen_titles)
                + "\n"
            )
        temp = min(0.98, base_temp + 0.04 * k)
        try:
            resp = claude_client.call(
                system=_FRESH_SYSTEM,
                user=prompt,
                max_tokens=max_tokens,
                temperature=temp,
                json_mode=True,
            )
        except Exception:
            continue
        if not getattr(resp, "success", False):
            continue
        parsed = _parse_idea_json(getattr(resp, "text", ""))
        if not parsed:
            continue
        idea = _fresh_dict_to_idea(parsed, topic)
        if idea is not None:
            out.append(idea)
            seen_titles.append(idea.title)

    return out
