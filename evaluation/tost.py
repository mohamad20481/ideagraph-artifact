"""
evaluation/tost.py — non-inferiority and equivalence testing (Welch TOST).

Used for the CrossVal-style comparison "are generated ideas non-inferior to
published ideas on dimension X, within margin delta?".

Statistical definitions (two independent samples, higher = better):
    d      = mean(test) - mean(ref)
    SE     = sqrt(s_t^2/n_t + s_r^2/n_r)                    (Welch)
    df     = Welch-Satterthwaite approximation
NON-INFERIORITY at margin delta > 0:
    H0: mu_t - mu_r <= -delta      H1: mu_t - mu_r > -delta
    t  = (d + delta) / SE          p = P(T_df > t)  (one-sided upper tail)
    non-inferiority is ESTABLISHED iff p < alpha.
EQUIVALENCE (TOST proper) at +/- delta:
    two one-sided tests; p_tost = max(p_lower, p_upper); equivalence iff
    p_tost < alpha. Reported alongside the conventional (1 - 2*alpha)
    confidence interval for d — equivalence holds iff that CI lies within
    (-delta, +delta).

Sanity anchors (used by the tests):
    d == -delta  ->  t = 0  ->  p = 0.5 exactly (never significant)
    d == 0 with tiny SE -> p ~ 0 (clearly non-inferior)

Also provides Clopper-Pearson intervals + an exact binomial test for judge
discriminability (n paired forced-choice judgments vs. chance = 0.5).

All functions raise EvaluationError on degenerate inputs (n < 2, zero
variance in both samples, delta <= 0) rather than returning NaN.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Sequence, Tuple

from evaluation import EvaluationError


def _clean(sample: Sequence[float], name: str):
    import numpy as np
    arr = np.asarray(list(sample), dtype="float64")
    if arr.size < 2:
        raise EvaluationError(f"{name}: need >= 2 observations, got {arr.size}")
    if not np.isfinite(arr).all():
        raise EvaluationError(f"{name}: contains NaN/inf")
    return arr


def _welch(test, ref) -> Tuple[float, float, float]:
    """(d, SE, df) via Welch. Raises if SE == 0 (both samples constant)."""
    import numpy as np
    d = float(test.mean() - ref.mean())
    vt, vr = test.var(ddof=1), ref.var(ddof=1)
    nt, nr = test.size, ref.size
    se2 = vt / nt + vr / nr
    if se2 <= 0.0:
        raise EvaluationError(
            "Welch SE is zero (both samples constant) — test undefined"
        )
    se = float(np.sqrt(se2))
    df = float(se2 ** 2 / ((vt / nt) ** 2 / (nt - 1) + (vr / nr) ** 2 / (nr - 1)))
    return d, se, df


@dataclass
class NonInferiorityResult:
    """One-sided Welch non-inferiority test at margin delta."""
    mean_test: float
    mean_ref: float
    diff: float                 # mean_test - mean_ref
    delta: float
    se: float
    df: float
    t_stat: float
    p_value: float              # one-sided
    alpha: float
    established: bool           # p < alpha
    ci_low: float               # (1 - 2*alpha) CI for diff — the TOST-standard CI
    ci_high: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def noninferiority_test(
    test: Sequence[float],
    ref: Sequence[float],
    delta: float,
    alpha: float = 0.05,
) -> NonInferiorityResult:
    """Is `test` non-inferior to `ref` within margin `delta` (higher=better)?"""
    if not (delta > 0):
        raise EvaluationError(f"delta must be > 0, got {delta}")
    if not (0 < alpha < 0.5):
        raise EvaluationError(f"alpha must be in (0, 0.5), got {alpha}")
    from scipy import stats
    t_arr, r_arr = _clean(test, "test"), _clean(ref, "ref")
    d, se, df = _welch(t_arr, r_arr)
    t_stat = (d + delta) / se
    p = float(stats.t.sf(t_stat, df))
    tcrit = float(stats.t.ppf(1.0 - alpha, df))   # (1-2a) CI uses alpha per tail
    return NonInferiorityResult(
        mean_test=float(t_arr.mean()), mean_ref=float(r_arr.mean()),
        diff=d, delta=float(delta), se=se, df=df,
        t_stat=float(t_stat), p_value=p, alpha=float(alpha),
        established=bool(p < alpha),
        ci_low=d - tcrit * se, ci_high=d + tcrit * se,
    )


@dataclass
class TostResult:
    """Two-one-sided-tests equivalence result at +/- delta."""
    diff: float
    delta: float
    se: float
    df: float
    p_lower: float              # H1: diff > -delta
    p_upper: float              # H1: diff < +delta
    p_tost: float               # max of the two
    alpha: float
    equivalent: bool            # p_tost < alpha
    ci_low: float               # (1 - 2*alpha) CI
    ci_high: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def tost_equivalence(
    test: Sequence[float],
    ref: Sequence[float],
    delta: float,
    alpha: float = 0.05,
) -> TostResult:
    """Full TOST: are the two means equivalent within +/- delta?"""
    if not (delta > 0):
        raise EvaluationError(f"delta must be > 0, got {delta}")
    if not (0 < alpha < 0.5):
        raise EvaluationError(f"alpha must be in (0, 0.5), got {alpha}")
    from scipy import stats
    t_arr, r_arr = _clean(test, "test"), _clean(ref, "ref")
    d, se, df = _welch(t_arr, r_arr)
    p_lower = float(stats.t.sf((d + delta) / se, df))   # reject diff <= -delta
    p_upper = float(stats.t.cdf((d - delta) / se, df))  # reject diff >= +delta
    p_tost = max(p_lower, p_upper)
    tcrit = float(stats.t.ppf(1.0 - alpha, df))
    return TostResult(
        diff=d, delta=float(delta), se=se, df=df,
        p_lower=p_lower, p_upper=p_upper, p_tost=p_tost,
        alpha=float(alpha), equivalent=bool(p_tost < alpha),
        ci_low=d - tcrit * se, ci_high=d + tcrit * se,
    )


# ── Discriminability (forced-choice vs. chance) ──────────────────────────────

def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided CI for a binomial proportion."""
    if n <= 0 or k < 0 or k > n:
        raise EvaluationError(f"clopper_pearson: invalid k={k}, n={n}")
    from scipy import stats
    low = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return low, high


@dataclass
class DiscriminabilityResult:
    n_judgments: int
    n_correct: int
    accuracy: float
    ci_low: float
    ci_high: float
    p_two_sided: float          # exact binomial vs. chance 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def discriminability(n_correct: int, n_judgments: int,
                     alpha: float = 0.05) -> DiscriminabilityResult:
    """Can judges tell generated from published better than chance?"""
    if n_judgments <= 0:
        raise EvaluationError("discriminability: n_judgments must be > 0")
    if not (0 <= n_correct <= n_judgments):
        raise EvaluationError(
            f"discriminability: n_correct={n_correct} outside [0, {n_judgments}]"
        )
    from scipy import stats
    res = stats.binomtest(n_correct, n_judgments, p=0.5, alternative="two-sided")
    low, high = clopper_pearson(n_correct, n_judgments, alpha=alpha)
    return DiscriminabilityResult(
        n_judgments=int(n_judgments), n_correct=int(n_correct),
        accuracy=n_correct / n_judgments,
        ci_low=low, ci_high=high, p_two_sided=float(res.pvalue),
    )
