"""
evaluation — the IdeaGraph evaluation harness.

Added 2026-07 in response to ARR reviewer feedback that the paper's metrics
(RAS, NFT, DI/Div-Pair, Nov-A, Feas-A) were not formally defined and that the
released artifact did not contain the evaluation code.

HONESTY NOTE (read this before citing any number):
    This package is a fresh implementation with explicitly chosen, documented
    definitions (see evaluation/README.md). Running it produces NEW results.
    Any numbers produced by this harness SUPERSEDE previously reported ones;
    they are not expected to reproduce any earlier table. Human-panel ratings
    are data, not code — this package defines their schema and analysis only.

Design contract (differs from the app's best-effort scorers ON PURPOSE):
    Evaluation code FAILS LOUDLY. Missing embedder, empty inputs, or a judge
    that returns garbage raise EvaluationError — they never silently score 0.
"""

class EvaluationError(RuntimeError):
    """Raised when an evaluation prerequisite is missing or an invariant is
    violated. Evaluation must never degrade silently."""


__all__ = ["EvaluationError"]
