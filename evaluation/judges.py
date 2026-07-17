"""
evaluation/judges.py — blind LLM judge panel for idea rating.

Produces the per-idea scores that feed Nov-A / Feas-A / NFT and the
per-dimension distributions that feed the TOST non-inferiority tests.

Protocol:
  * Judges see ONLY a blinded 7-field tuple (see crossval.blind) — no source,
    no provenance, no title.
  * Six dimensions, each on an anchored 1-5 scale (anchors below). Judging
    temperature is 0.0.
  * A panel is a list of independently-callable judge clients (ideally from
    DIFFERENT providers). Scores are averaged across judges per dimension.
  * Unit mapping for downstream metrics: score_unit = (score - 1) / 4, so
    1 -> 0.0 and 5 -> 1.0. Nov-A = mean unit novelty; Feas-A = mean unit
    feasibility (means over ideas of the judge-averaged per-idea scores).

Failure policy: a judge call that fails or returns unparseable/out-of-range
JSON raises EvaluationError. Evaluation runs must be complete-or-failed,
never silently partial. (Callers may retry; they may not skip.)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from evaluation import EvaluationError

DIMENSIONS = (
    "novelty", "feasibility", "clarity", "significance", "excitement", "overall",
)

RUBRIC = """\
Rate the research idea on SIX dimensions, each an integer 1-5:

novelty      1=well-known/already published as stated .. 3=solid increment .. 5=genuinely new framing or method
feasibility  1=cannot be executed as described .. 3=executable with substantial effort/resources .. 5=straightforwardly executable by a typical lab
clarity      1=cannot tell what is proposed .. 3=understandable with effort .. 5=precise and unambiguous
significance 1=consequence-free .. 3=useful to its subfield .. 5=would change how the area works
excitement   1=would not read the paper .. 3=would skim it .. 5=would drop what I'm doing to read it
overall      1=reject-tier .. 3=borderline .. 5=award-tier
"""

_JUDGE_SYSTEM = (
    "You are an expert reviewer for a top AI venue, rating research ideas "
    "presented as structured tuples. You do not know whether an idea is "
    "machine-generated or from a published paper — rate purely on content. "
    "Be calibrated and use the full scale. Output ONLY valid JSON."
)


def _judge_user_prompt(item: Dict[str, Any]) -> str:
    tuple_block = "\n".join(
        f"  {k}: {item[k]}"
        for k in ("motivation", "hypothesis", "method_sketch", "dataset",
                  "metrics", "baselines", "expected_outcome")
        if item.get(k)
    )
    return (
        f"TOPIC: {item.get('topic', '(unspecified)')}\n\n"
        f"IDEA TUPLE:\n{tuple_block}\n\n"
        f"{RUBRIC}\n"
        "Return JSON exactly:\n"
        '{"novelty": n, "feasibility": n, "clarity": n, '
        '"significance": n, "excitement": n, "overall": n}'
    )


def _parse_scores(text: str, who: str, blind_id: str) -> Dict[str, int]:
    if not text:
        raise EvaluationError(f"judge {who} on {blind_id}: empty response")
    s = text.strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    parsed = None
    try:
        parsed = json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                parsed = None
    if not isinstance(parsed, dict):
        raise EvaluationError(
            f"judge {who} on {blind_id}: unparseable response: {text[:120]!r}"
        )
    out: Dict[str, int] = {}
    for dim in DIMENSIONS:
        if dim not in parsed:
            raise EvaluationError(f"judge {who} on {blind_id}: missing '{dim}'")
        try:
            v = int(round(float(parsed[dim])))
        except (TypeError, ValueError) as e:
            raise EvaluationError(
                f"judge {who} on {blind_id}: non-numeric '{dim}': {parsed[dim]!r}"
            ) from e
        if not (1 <= v <= 5):
            raise EvaluationError(
                f"judge {who} on {blind_id}: '{dim}'={v} outside 1-5"
            )
        out[dim] = v
    return out


@dataclass
class Judge:
    """One panel member: a name + a client with the project-standard
    .call(system, user, max_tokens, temperature, json_mode) interface
    returning an object with .success and .text."""
    name: str
    client: Any
    max_tokens: int = 300

    def rate(self, item: Dict[str, Any], retries: int = 1) -> Dict[str, int]:
        blind_id = str(item.get("blind_id", "?"))
        last_err: Optional[Exception] = None
        for attempt in range(1 + max(0, retries)):
            # Reasoning models can exhaust the output budget on hidden
            # reasoning ("empty response"). Retrying at the same budget
            # fails identically, so escalate: ×2 per attempt, capped.
            eff_tokens = min(int(self.max_tokens * (2 ** attempt)), 16000)
            try:
                resp = self.client.call(
                    system=_JUDGE_SYSTEM,
                    user=_judge_user_prompt(item),
                    max_tokens=eff_tokens,
                    temperature=0.0,
                    json_mode=True,
                )
            except Exception as e:
                last_err = e
                continue
            if not getattr(resp, "success", False):
                last_err = EvaluationError(
                    f"judge {self.name} on {blind_id}: call failed: "
                    f"{getattr(resp, 'text', '')[:120]}"
                )
                continue
            try:
                return _parse_scores(getattr(resp, "text", ""), self.name, blind_id)
            except EvaluationError as e:
                last_err = e
                continue
        raise EvaluationError(
            f"judge {self.name} on {blind_id}: exhausted retries: {last_err}"
        )


@dataclass
class PanelRatings:
    """All ratings for one blinded benchmark view."""
    per_item: Dict[str, Dict[str, Dict[str, int]]] = field(default_factory=dict)
    # blind_id -> judge_name -> {dimension: 1..5}

    def judge_names(self) -> List[str]:
        names: List[str] = []
        for ratings in self.per_item.values():
            for n in ratings:
                if n not in names:
                    names.append(n)
        return names

    def mean_scores(self, blind_id: str) -> Dict[str, float]:
        """Judge-averaged 1-5 score per dimension for one item."""
        ratings = self.per_item.get(blind_id)
        if not ratings:
            raise EvaluationError(f"mean_scores: no ratings for {blind_id!r}")
        out: Dict[str, float] = {}
        for dim in DIMENSIONS:
            vals = [r[dim] for r in ratings.values()]
            out[dim] = sum(vals) / len(vals)
        return out

    def unit_scores(self, blind_id: str) -> Dict[str, float]:
        """Same, mapped to [0,1] via (s - 1) / 4 — feeds Nov-A/Feas-A/NFT."""
        return {d: (s - 1.0) / 4.0 for d, s in self.mean_scores(blind_id).items()}

    def dimension_sample(self, blind_ids: Sequence[str], dim: str) -> List[float]:
        """The per-idea judge-averaged scores for one dimension (1-5 scale) —
        the sample that goes into the TOST tests."""
        if dim not in DIMENSIONS:
            raise EvaluationError(f"dimension_sample: unknown dimension {dim!r}")
        return [self.mean_scores(b)[dim] for b in blind_ids]

    def to_jsonable(self) -> Dict[str, Any]:
        return {"per_item": self.per_item}


def rate_all(
    blinded_view: Sequence[Dict[str, Any]],
    judges: Sequence[Judge],
    on_progress: Any = None,
) -> PanelRatings:
    """Every judge rates every blinded item. Complete-or-raise."""
    if not blinded_view:
        raise EvaluationError("rate_all: empty benchmark view")
    if not judges:
        raise EvaluationError("rate_all: no judges")
    names = [j.name for j in judges]
    if len(set(names)) != len(names):
        raise EvaluationError(f"rate_all: duplicate judge names: {names}")
    ratings = PanelRatings()
    total = len(blinded_view) * len(judges)
    done = 0
    for item in blinded_view:
        blind_id = str(item.get("blind_id") or "")
        if not blind_id:
            raise EvaluationError("rate_all: item missing blind_id")
        ratings.per_item[blind_id] = {}
        for judge in judges:
            ratings.per_item[blind_id][judge.name] = judge.rate(item)
            done += 1
            if on_progress:
                try:
                    on_progress(f"judging {done}/{total} ({judge.name} on {blind_id})")
                except Exception:
                    pass
    return ratings


_PING_ITEM = {
    "blind_id": "ping-000", "topic": "connectivity self-test (synthetic)",
    "motivation": "verify the judge endpoint returns rubric JSON",
    "hypothesis": "the endpoint responds with six 1-5 integers",
    "method_sketch": "send this synthetic tuple and parse the response",
    "dataset": "none (synthetic ping)", "metrics": "response validity",
    "baselines": "none", "expected_outcome": "a parseable rating",
}


def build_panel(specs: List[str], ping: bool = True) -> List[Judge]:
    """Build a judge panel from explicit provider specs.

    Each spec is "provider" or "provider:model" over the project's
    OpenAI-compatible providers (deepseek, kimi, openai, groq, xai,
    gemini, azure). Distinct providers are strongly preferred; a panel of
    distinct MODELS from one provider is acceptable but the paper must
    report the panel composition exactly as used.

    With ping=True every judge must successfully rate a synthetic tuple
    before the panel is returned — a panel is verified or it is refused.
    """
    from claude_provider import OpenAICompatClient, _provider_credentials
    if not specs:
        raise EvaluationError("build_panel: no specs")
    judges: List[Judge] = []
    problems: List[str] = []
    for spec in specs:
        provider, _, model = spec.partition(":")
        provider = provider.strip().lower()
        model = model.strip()
        api_key, base_url, default_model = _provider_credentials(provider)
        if not api_key or not base_url:
            problems.append(f"{spec}: no credentials configured")
            continue
        client = OpenAICompatClient(
            api_key=api_key, model=model or default_model,
            base_url=base_url, provider_name=provider,
        )
        # Reasoning models (Kimi K2.x etc.) consume output budget on hidden
        # reasoning before emitting content — a 300-token cap yields empty
        # responses. Give them room; the rubric JSON itself stays tiny.
        eff_model = (model or default_model).lower()
        max_toks = 4000 if eff_model.startswith(("kimi-k2", "k2")) else 300
        judges.append(Judge(name=spec.replace(":", "/"), client=client,
                            max_tokens=max_toks))
    if problems:
        raise EvaluationError("build_panel: " + "; ".join(problems))
    if ping:
        for j in judges:
            try:
                j.rate(dict(_PING_ITEM), retries=1)
            except EvaluationError as e:
                problems.append(f"{j.name}: ping failed ({e})")
        if problems:
            raise EvaluationError("build_panel: " + "; ".join(problems))
    return judges


def default_panel() -> List[Judge]:
    """Best-available real panel from the project's configured providers.
    Prefers 3 distinct providers via multi_llm_ensemble when configured;
    raises if fewer than ONE working client exists (a 1-judge panel is
    allowed but the paper must report the panel composition truthfully)."""
    judges: List[Judge] = []
    try:
        from claude_provider import get_claude_client
        c = get_claude_client()
        if c is not None:
            judges.append(Judge(name="primary", client=c))
    except Exception:
        pass
    if not judges:
        raise EvaluationError(
            "default_panel: no working LLM client configured — set up a "
            "provider before running judged evaluation"
        )
    return judges
