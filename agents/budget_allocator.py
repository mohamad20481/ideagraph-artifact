"""
agents/budget_allocator.py - Adaptive budget allocation across pipeline stages.

Dynamically allocates API budget based on which stages are producing
the most value. Stages that generate higher quality outputs get more budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class StageMetrics:
    """Track performance of a pipeline stage."""
    name: str
    calls: int = 0
    tokens_used: int = 0
    quality_sum: float = 0.0
    failures: int = 0

    @property
    def avg_quality(self) -> float:
        return self.quality_sum / max(self.calls, 1)

    @property
    def success_rate(self) -> float:
        return (self.calls - self.failures) / max(self.calls, 1)

    @property
    def efficiency(self) -> float:
        """Quality per 1000 tokens."""
        return (self.quality_sum / max(self.tokens_used, 1)) * 1000


class AdaptiveBudgetAllocator:
    """
    Dynamically allocate API budget across pipeline stages.

    Strategy: stages with higher quality output and better efficiency
    get proportionally more budget in subsequent iterations.
    """

    STAGES = [
        "ideation", "experiment_design", "tree_search",
        "code_generation", "analysis", "paper_writing", "review",
    ]

    # Base allocation (percentage of total budget per stage)
    BASE_ALLOCATION = {
        "ideation": 0.30,
        "experiment_design": 0.10,
        "tree_search": 0.10,
        "code_generation": 0.15,
        "analysis": 0.10,
        "paper_writing": 0.15,
        "review": 0.10,
    }

    def __init__(self, total_budget_usd: float):
        self.total_budget = total_budget_usd
        self.spent = 0.0
        self.metrics: Dict[str, StageMetrics] = {
            s: StageMetrics(name=s) for s in self.STAGES
        }
        self._allocation = dict(self.BASE_ALLOCATION)
        # Avoid summing all StageMetrics.calls on every record (was O(stages)
        # per call, then mod-2 → rebalance every 2 calls). Track total calls
        # incrementally and use exponential backoff so rebalances are rare in
        # long pipelines (4, 8, 16, 32, … calls instead of 2, 4, 6, 8, …).
        self._total_calls = 0
        self._next_rebalance = 4

    def get_stage_budget(self, stage: str) -> float:
        """Get allocated budget for a specific stage."""
        remaining = self.total_budget - self.spent
        fraction = self._allocation.get(stage, 0.1)
        return remaining * fraction

    def record_stage_result(
        self, stage: str, tokens_used: int, quality: float, failed: bool = False,
    ) -> None:
        """Record a stage execution result for adaptive learning."""
        m = self.metrics.get(stage)
        if not m:
            return
        m.calls += 1
        m.tokens_used += tokens_used
        m.quality_sum += quality
        if failed:
            m.failures += 1

        # Exponential-backoff rebalance: re-evaluate at calls 4, 8, 16, 32, ...
        # Allocation stabilises quickly; the late-pipeline rebalances are
        # mostly redundant CPU.
        self._total_calls += 1
        if self._total_calls >= self._next_rebalance:
            self._rebalance()
            self._next_rebalance *= 2

    def record_spend(self, amount_usd: float) -> None:
        """Record actual spending."""
        self.spent += amount_usd

    def _rebalance(self) -> None:
        """Rebalance allocation based on observed performance."""
        # Compute efficiency scores
        efficiency = {}
        for name, m in self.metrics.items():
            if m.calls > 0:
                # Weighted: 60% quality, 40% efficiency
                score = 0.6 * m.avg_quality + 0.4 * min(m.efficiency, 1.0)
                # Penalize high failure rates
                score *= m.success_rate
                efficiency[name] = max(score, 0.05)
            else:
                efficiency[name] = self.BASE_ALLOCATION.get(name, 0.1)

        # Normalize to sum to 1.0
        total = sum(efficiency.values())
        if total > 0:
            # Blend: 50% adaptive + 50% base (avoid extreme shifts)
            for name in self.STAGES:
                adaptive = efficiency.get(name, 0.1) / total
                base = self.BASE_ALLOCATION.get(name, 0.1)
                self._allocation[name] = 0.5 * adaptive + 0.5 * base

    @property
    def remaining_budget(self) -> float:
        return max(0, self.total_budget - self.spent)

    @property
    def budget_used_pct(self) -> float:
        return (self.spent / self.total_budget * 100) if self.total_budget > 0 else 0

    def summary(self) -> Dict:
        """Get allocation summary."""
        return {
            "total_budget": self.total_budget,
            "spent": round(self.spent, 4),
            "remaining": round(self.remaining_budget, 4),
            "allocation": {k: round(v, 3) for k, v in self._allocation.items()},
            "stage_metrics": {
                k: {
                    "calls": v.calls, "avg_quality": round(v.avg_quality, 3),
                    "success_rate": round(v.success_rate, 3),
                }
                for k, v in self.metrics.items() if v.calls > 0
            },
        }
