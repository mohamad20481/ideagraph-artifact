"""
genetic_ideation.py — evolutionary novelty via crossover + mutate.

Treat ideas as genomes; pair-wise crossover splices method-from-A with
hypothesis-from-B (and the LLM rewrites coherently); mutation perturbs
one or two fields; selection keeps the top half by fitness. Iterate over
N generations. Produces ideas the LLM alone wouldn't propose because the
*structure* is recombined, not generated.

Fitness is `quality_score` if probes have populated it, else a novelty
heuristic from token-coverage against the population. Pluggable via the
`fitness_fn` argument.

Public API:
    EvolutionResult                                       → dataclass
    crossover(parent_a, parent_b, claude_client=None)     → Optional[Idea]
    mutate(idea, claude_client=None, mutation_strength=…) → Optional[Idea]
    evolve(initial_population, n_generations=3, ...)      → EvolutionResult
"""
from __future__ import annotations

import json
import re
import random as _random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


_AUTOLOAD = object()


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvolutionResult:
    """Outcome of running the genetic algorithm over N generations."""
    initial_size: int = 0
    n_generations: int = 0
    final_population: List[Idea] = field(default_factory=list)
    fitness_history: List[List[float]] = field(default_factory=list)
    crossover_count: int = 0
    mutation_count: int = 0
    elapsed_s: float = 0.0

    def summary(self) -> str:
        if not self.fitness_history:
            return "(no generations)"
        last = self.fitness_history[-1] if self.fitness_history else []
        best = max(last) if last else 0.0
        return (
            f"GA: {len(self.final_population)} survivors after "
            f"{self.n_generations} generations · "
            f"crossovers={self.crossover_count} · "
            f"mutations={self.mutation_count} · "
            f"best fitness={best:.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Crossover
# ─────────────────────────────────────────────────────────────────────────────

_CROSSOVER_SYSTEM = (
    "You are a research scientist combining elements of two parent ideas "
    "into one coherent offspring. You will be given two parents and a "
    "splice plan (which fields to inherit from which parent). Produce "
    "ONE offspring that integrates them coherently — not a Frankenstein "
    "concatenation. Rewrite text so the offspring reads as a single "
    "research proposal. Output ONLY valid JSON. methodology_type must be "
    f"one of: {', '.join(METHODOLOGY_TYPES)}. novelty_level must be one "
    f"of: {', '.join(NOVELTY_LEVELS)}."
)


def _parent_summary(idea: Any) -> Dict[str, str]:
    d = idea.to_dict() if hasattr(idea, "to_dict") else (
        idea if isinstance(idea, dict) else {}
    )
    return {
        "title": str(d.get("title", "")),
        "motivation": str(d.get("motivation", ""))[:400],
        "method": str(d.get("method", ""))[:600],
        "hypothesis": str(d.get("hypothesis", ""))[:300],
        "methodology_type": d.get("methodology_type") or "?",
        "novelty_level": d.get("novelty_level") or "?",
    }


def _crossover_user_prompt(a: Any, b: Any, plan: Dict[str, str]) -> str:
    pa, pb = _parent_summary(a), _parent_summary(b)
    plan_str = "\n".join(f"  • {field}: from {src}" for field, src in plan.items())
    return (
        f"### Parent A\n"
        f"Title: {pa['title']}\n"
        f"Motivation: {pa['motivation']}\n"
        f"Method: {pa['method']}\n"
        f"Hypothesis: {pa['hypothesis']}\n"
        f"methodology_type: {pa['methodology_type']}\n"
        f"novelty_level: {pa['novelty_level']}\n\n"
        f"### Parent B\n"
        f"Title: {pb['title']}\n"
        f"Motivation: {pb['motivation']}\n"
        f"Method: {pb['method']}\n"
        f"Hypothesis: {pb['hypothesis']}\n"
        f"methodology_type: {pb['methodology_type']}\n"
        f"novelty_level: {pb['novelty_level']}\n\n"
        f"### Splice plan\n{plan_str}\n\n"
        f"### Instructions\n"
        f"Produce ONE coherent offspring idea following the splice plan. "
        f"Where the plan says 'from A', inherit that field's spirit from "
        f"parent A; same for B. Then REWRITE the combined idea so it "
        f"reads as a single coherent proposal — not a copy-paste mashup. "
        f"In lineage_note, state which elements you took from which parent.\n\n"
        f"Return JSON:\n"
        "{\n"
        '  "title": "<offspring title>",\n'
        '  "motivation": "<unified motivation>",\n'
        '  "method": "<coherent method>",\n'
        '  "hypothesis": "<falsifiable claim>",\n'
        '  "resources": "<resources needed>",\n'
        '  "expected_outcome": "<measurable outcome>",\n'
        '  "risk_assessment": "<main risks>",\n'
        '  "source_strategy": "G",\n'
        f'  "methodology_type": "<one of: {", ".join(METHODOLOGY_TYPES)}>",\n'
        f'  "novelty_level": "<one of: {", ".join(NOVELTY_LEVELS)}>",\n'
        '  "lineage_note": "<one sentence: which fields came from '
        'parent A vs B and how you reconciled them>"\n'
        "}"
    )


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
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


def _idea_from_dict(parsed: Dict[str, Any], parents: Tuple[Any, Any],
                      strategy_code: str, mode: str,
                      generation: int) -> Optional[Idea]:
    if not all(str(parsed.get(k, "")).strip() for k in ("title", "method", "hypothesis")):
        return None
    method_type = parsed.get("methodology_type") or ""
    if method_type not in METHODOLOGY_TYPES:
        method_type = None
    novelty = parsed.get("novelty_level") or ""
    if novelty not in NOVELTY_LEVELS:
        novelty = None
    idea = Idea(
        title=str(parsed.get("title", ""))[:200],
        motivation=str(parsed.get("motivation", ""))[:1000],
        method=str(parsed.get("method", ""))[:2000],
        hypothesis=str(parsed.get("hypothesis", ""))[:1000],
        resources=str(parsed.get("resources", ""))[:500],
        expected_outcome=str(parsed.get("expected_outcome", ""))[:500],
        risk_assessment=str(parsed.get("risk_assessment", ""))[:500],
        source_strategy=strategy_code,
        methodology_type=method_type,
        novelty_level=novelty,
        generation=generation,
        parent_title=(
            _parent_summary(parents[0])["title"] if parents and parents[0] else None
        ),
    )
    idea.execution_meta = {
        "lineage_note": str(parsed.get("lineage_note", ""))[:400],
        "regen_mode": mode,
        "parent_a_title": _parent_summary(parents[0])["title"] if parents[0] else "",
        "parent_b_title": _parent_summary(parents[1])["title"] if (
            len(parents) > 1 and parents[1]
        ) else "",
    }
    return idea


def crossover(
    parent_a: Any,
    parent_b: Any,
    claude_client: Any = _AUTOLOAD,
    rng: Optional[_random.Random] = None,
    max_tokens: int = 900,
    temperature: float = 0.75,
) -> Optional[Idea]:
    """Splice two parents into one offspring via LLM rewriting.

    The splice plan randomizes which field is inherited from which parent,
    so two crossover calls on the same pair don't return the same offspring.
    """
    if rng is None:
        rng = _random
    if claude_client is _AUTOLOAD:
        try:
            from claude_provider import get_claude_client
            claude_client = get_claude_client()
        except Exception:
            claude_client = None
    if claude_client is None:
        return None

    plan = {
        "motivation": rng.choice(["A", "B"]),
        "method": rng.choice(["A", "B"]),
        "hypothesis": rng.choice(["A", "B"]),
        "methodology_type": rng.choice(["A", "B"]),
    }
    # Avoid plans that inherit everything from one parent (that's not crossover)
    if all(v == plan["method"] for v in plan.values()):
        plan["hypothesis"] = "A" if plan["hypothesis"] == "B" else "B"

    try:
        resp = claude_client.call(
            system=_CROSSOVER_SYSTEM,
            user=_crossover_user_prompt(parent_a, parent_b, plan),
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=True,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return None

    parent_gen = 0
    try:
        pa = parent_a.to_dict() if hasattr(parent_a, "to_dict") else parent_a
        pb = parent_b.to_dict() if hasattr(parent_b, "to_dict") else parent_b
        parent_gen = max(int(pa.get("generation") or 0),
                          int(pb.get("generation") or 0))
    except Exception:
        parent_gen = 0

    return _idea_from_dict(
        parsed, (parent_a, parent_b),
        strategy_code="G", mode="genetic_crossover",
        generation=parent_gen + 1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mutation
# ─────────────────────────────────────────────────────────────────────────────

_MUTATE_SYSTEM = (
    "You are a research scientist mutating an idea — perturbing one or "
    "two components while keeping the overall direction. The mutation "
    "should produce a genuinely different idea, not just a paraphrase. "
    "Output ONLY valid JSON with the idea schema. methodology_type must "
    f"be one of: {', '.join(METHODOLOGY_TYPES)}. novelty_level must be "
    f"one of: {', '.join(NOVELTY_LEVELS)}."
)

_MUTATION_KINDS = [
    "swap the dataset for a related but distinct one",
    "replace one architectural choice with an alternative",
    "shift the scale up or down by an order of magnitude",
    "change one key assumption in the hypothesis",
    "switch from supervised to self-supervised (or vice versa)",
    "add a new evaluation axis the original idea ignores",
    "replace a learned component with an analytic one (or vice versa)",
]


def _mutate_user_prompt(idea: Any, rng: _random.Random,
                          mutation_strength: float) -> Tuple[str, str]:
    d = _parent_summary(idea)
    n_mutations = 1 if mutation_strength < 0.6 else 2
    chosen_kinds = rng.sample(_MUTATION_KINDS, k=min(n_mutations, len(_MUTATION_KINDS)))
    kinds_str = "\n".join(f"  • {k}" for k in chosen_kinds)
    return chosen_kinds[0], (
        f"### Original idea\n"
        f"Title: {d['title']}\n"
        f"Motivation: {d['motivation']}\n"
        f"Method: {d['method']}\n"
        f"Hypothesis: {d['hypothesis']}\n"
        f"methodology_type: {d['methodology_type']}\n"
        f"novelty_level: {d['novelty_level']}\n\n"
        f"### Mutation(s) to apply\n{kinds_str}\n\n"
        f"### Instructions\n"
        f"Apply the mutation(s) above. Keep title and motivation close to "
        f"the original; CHANGE the method or hypothesis to reflect the "
        f"mutation. State explicitly in mutation_note which mutation(s) "
        f"you applied and how.\n\n"
        f"Return JSON:\n"
        "{\n"
        '  "title": "<mutated title; near-original>",\n'
        '  "motivation": "<near-original motivation>",\n'
        '  "method": "<mutated method>",\n'
        '  "hypothesis": "<mutated hypothesis>",\n'
        '  "resources": "<resources needed>",\n'
        '  "expected_outcome": "<measurable result>",\n'
        '  "risk_assessment": "<main risks>",\n'
        '  "source_strategy": "G",\n'
        f'  "methodology_type": "<one of: {", ".join(METHODOLOGY_TYPES)}>",\n'
        f'  "novelty_level": "<one of: {", ".join(NOVELTY_LEVELS)}>",\n'
        '  "mutation_note": "<one sentence: which mutation you applied>"\n'
        "}"
    )


def mutate(
    idea: Any,
    claude_client: Any = _AUTOLOAD,
    rng: Optional[_random.Random] = None,
    mutation_strength: float = 0.5,
    max_tokens: int = 900,
    temperature: float = 0.7,
) -> Optional[Idea]:
    """Apply 1-2 mutations to an idea via LLM rewriting."""
    if rng is None:
        rng = _random
    if claude_client is _AUTOLOAD:
        try:
            from claude_provider import get_claude_client
            claude_client = get_claude_client()
        except Exception:
            claude_client = None
    if claude_client is None:
        return None

    mutation_kind, prompt = _mutate_user_prompt(idea, rng, mutation_strength)
    try:
        resp = claude_client.call(
            system=_MUTATE_SYSTEM, user=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=True,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return None

    parent_d = idea.to_dict() if hasattr(idea, "to_dict") else idea
    parent_gen = int(parent_d.get("generation") or 0)
    mutant = _idea_from_dict(
        parsed, (idea, None),
        strategy_code="G", mode="genetic_mutation",
        generation=parent_gen + 1,
    )
    if mutant is not None:
        if mutant.execution_meta is None:
            mutant.execution_meta = {}
        mutant.execution_meta["mutation_note"] = str(
            parsed.get("mutation_note", "")
        )[:400]
        mutant.execution_meta["mutation_kind"] = mutation_kind
    return mutant


# ─────────────────────────────────────────────────────────────────────────────
# Fitness + selection + evolve loop
# ─────────────────────────────────────────────────────────────────────────────

def _default_fitness(idea: Any, population: List[Any]) -> float:
    """If probe quality is set, use it (with a small novelty bonus relative
    to the population). Otherwise fall back to a pure token-rarity score."""
    d = idea.to_dict() if hasattr(idea, "to_dict") else idea
    q = float(d.get("quality_score") or 0.0)

    # Token rarity novelty bonus
    from collections import Counter
    _word_re = re.compile(r"[A-Za-z]{4,}")

    def _toks(x):
        d2 = x.to_dict() if hasattr(x, "to_dict") else x
        text = (str(d2.get("title", "")) + " "
                + str(d2.get("method", ""))).lower()
        return set(_word_re.findall(text))

    pop_counter: Counter = Counter()
    for other in population:
        if other is idea:
            continue
        for t in _toks(other):
            pop_counter[t] += 1
    my_toks = _toks(idea)
    n_pop = max(1, len(population) - 1)
    rarity = sum(1 - (pop_counter.get(t, 0) / n_pop) for t in my_toks)
    rarity_norm = rarity / max(1, len(my_toks))

    # Combine: 70% probe quality, 30% rarity bonus
    return 0.70 * q + 0.30 * rarity_norm


def evolve(
    initial_population: List[Any],
    n_generations: int = 3,
    claude_client: Any = _AUTOLOAD,
    crossover_rate: float = 0.6,
    mutation_rate: float = 0.3,
    elite_keep: int = 2,
    fitness_fn: Optional[Callable[[Any, List[Any]], float]] = None,
    seed: Optional[int] = None,
) -> EvolutionResult:
    """Run a small genetic algorithm over the population.

    Each generation:
      1. Score everyone via fitness_fn (defaults to quality + rarity bonus)
      2. Keep `elite_keep` highest-fitness as-is
      3. Fill the rest of the slots by crossover + mutation
      4. Repeat for n_generations

    Returns the final population sorted by fitness desc.
    """
    if not initial_population:
        return EvolutionResult(initial_size=0)
    if n_generations <= 0:
        return EvolutionResult(
            initial_size=len(initial_population),
            final_population=list(initial_population),
        )
    if claude_client is _AUTOLOAD:
        try:
            from claude_provider import get_claude_client
            claude_client = get_claude_client()
        except Exception:
            claude_client = None
    if claude_client is None:
        return EvolutionResult(
            initial_size=len(initial_population),
            final_population=list(initial_population),
        )

    fitness_fn = fitness_fn or _default_fitness
    rng = _random.Random(seed)

    import time
    started = time.time()
    population: List[Any] = list(initial_population)
    pop_size = len(population)
    history: List[List[float]] = []
    n_cross = 0
    n_mut = 0

    for gen in range(n_generations):
        # Score
        scored = [(p, fitness_fn(p, population)) for p in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        history.append([s for _, s in scored])

        if gen == n_generations - 1:
            # Last generation — return ranked population
            break

        # Keep elite
        keep = max(1, min(int(elite_keep), pop_size - 1))
        new_population: List[Any] = [p for p, _ in scored[:keep]]

        # Fill the rest by crossover + mutation
        while len(new_population) < pop_size:
            r = rng.random()
            if r < crossover_rate and len(scored) >= 2:
                # Tournament-style: pick from top half
                top_half = scored[: max(2, len(scored) // 2)]
                a = rng.choice(top_half)[0]
                b = rng.choice(top_half)[0]
                if a is b:
                    continue
                child = crossover(a, b, claude_client=claude_client, rng=rng)
                if child is not None:
                    new_population.append(child)
                    n_cross += 1
            elif r < crossover_rate + mutation_rate:
                parent = rng.choice(scored[: max(2, len(scored) // 2)])[0]
                m = mutate(parent, claude_client=claude_client, rng=rng)
                if m is not None:
                    new_population.append(m)
                    n_mut += 1
            else:
                # Carry-forward (no-op)
                parent = rng.choice(scored)[0]
                new_population.append(parent)
        population = new_population

    # Final ranking
    scored_final = [(p, fitness_fn(p, population)) for p in population]
    scored_final.sort(key=lambda x: x[1], reverse=True)
    if not history or len(history) < n_generations:
        history.append([s for _, s in scored_final])

    return EvolutionResult(
        initial_size=len(initial_population),
        n_generations=n_generations,
        final_population=[p for p, _ in scored_final],
        fitness_history=history,
        crossover_count=n_cross,
        mutation_count=n_mut,
        elapsed_s=time.time() - started,
    )
