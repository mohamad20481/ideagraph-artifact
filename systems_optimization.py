"""
systems_optimization.py - Distributed systems & reliability engineering
optimization for IdeaGraph.

Layer 11: Techniques from distributed consensus, reliability engineering,
queueing theory, control systems, compiler optimization, and network
science. Production-grade patterns for robust, scalable pipelines.

  1.  RaftConsensus            — Distributed log consensus for multi-agent agreement
  2.  BackpressureController   — Flow control to prevent stage overload
  3.  BulkheadIsolator         — Failure isolation between pipeline partitions
  4.  LoadShedder              — Graceful degradation under budget pressure
  5.  SlidingWindowRateLimiter — Token bucket rate limiting per stage
  6.  ConstantWorkInProgress   — WIP-limited Kanban scheduling for stages
  7.  CriticalPathScheduler    — CPM-based parallel stage scheduling
  8.  LoopInvariantHoister     — Cache invariant computations outside loops
  9.  BetweennessCentrality    — Network bottleneck detection in stage graph
  10. PIDController            — Closed-loop feedback control for quality targeting
  11. ExponentialBackoffPool   — Coordinated backoff across all agents
  12. CheckpointRecovery       — Transactional stage execution with rollback
"""

from __future__ import annotations

import hashlib
import math
import random
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ============================================================================
# 1. Raft Consensus (Simplified)
# ============================================================================

class RaftConsensus:
    """
    Simplified Raft-inspired consensus for multi-agent agreement.

    Models agents as nodes in a consensus group. One agent is elected
    leader; others are followers. The leader proposes values; followers
    vote. A value is committed when a majority agrees.

    Simplified from full Raft: no log replication, no term elections,
    just leader-based proposal → majority vote → commit.
    """

    @dataclass
    class Proposal:
        id: str
        value: Any
        proposer: str
        votes_for: Set[str] = field(default_factory=set)
        votes_against: Set[str] = field(default_factory=set)
        status: str = "pending"  # pending, committed, rejected
        timestamp: float = field(default_factory=time.time)

    def __init__(self):
        self._members: Set[str] = set()
        self._leader: Optional[str] = None
        self._proposals: Dict[str, "RaftConsensus.Proposal"] = {}
        self._committed: List[Any] = []
        self._lock = threading.Lock()

    def add_member(self, agent_id: str) -> None:
        with self._lock:
            self._members.add(agent_id)
            if self._leader is None:
                self._leader = agent_id

    def elect_leader(self) -> str:
        """Simple leader election: random member."""
        with self._lock:
            if self._members:
                self._leader = random.choice(sorted(self._members))
            return self._leader or ""

    def propose(self, value: Any, proposer: str = "") -> str:
        """Leader proposes a value. Returns proposal ID."""
        prop_id = hashlib.md5(f"{time.time()}{value}".encode(), usedforsecurity=False).hexdigest()[:8]
        with self._lock:
            self._proposals[prop_id] = self.Proposal(
                id=prop_id, value=value,
                proposer=proposer or self._leader or "",
            )
        return prop_id

    def vote(self, proposal_id: str, agent_id: str, approve: bool) -> None:
        with self._lock:
            prop = self._proposals.get(proposal_id)
            if not prop or prop.status != "pending":
                return
            if approve:
                prop.votes_for.add(agent_id)
            else:
                prop.votes_against.add(agent_id)
            # Check majority
            majority = len(self._members) // 2 + 1
            if len(prop.votes_for) >= majority:
                prop.status = "committed"
                self._committed.append(prop.value)
            elif len(prop.votes_against) >= majority:
                prop.status = "rejected"

    def get_committed(self) -> List[Any]:
        with self._lock:
            return list(self._committed)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "members": len(self._members),
                "leader": self._leader,
                "proposals": len(self._proposals),
                "committed": len(self._committed),
            }


# ============================================================================
# 2. Backpressure Controller
# ============================================================================

class BackpressureController:
    """
    Flow control to prevent downstream stage overload.

    When a stage's queue fills up (it can't process fast enough),
    backpressure signals upstream stages to slow down. Prevents
    wasted work and budget on items that will be dropped.

    Signal: pressure = queue_depth / max_queue_depth (0 → 1)
    Action: upstream production rate × (1 - pressure)
    """

    def __init__(self, max_queue_depth: int = 10):
        self.max_depth = max_queue_depth
        self._queues: Dict[str, int] = defaultdict(int)
        self._pressure: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def enqueue(self, stage: str) -> bool:
        """Try to enqueue an item. Returns False if backpressure is too high."""
        with self._lock:
            if self._queues[stage] >= self.max_depth:
                self._pressure[stage] = 1.0
                return False
            self._queues[stage] += 1
            self._pressure[stage] = self._queues[stage] / self.max_depth
            return True

    def dequeue(self, stage: str) -> None:
        with self._lock:
            self._queues[stage] = max(0, self._queues[stage] - 1)
            self._pressure[stage] = self._queues[stage] / self.max_depth

    def get_pressure(self, stage: str) -> float:
        """Get backpressure level (0=free, 1=full)."""
        with self._lock:
            return self._pressure.get(stage, 0.0)

    def should_produce(self, upstream_stage: str, downstream_stage: str) -> bool:
        """Should upstream produce more items?"""
        return self.get_pressure(downstream_stage) < 0.8

    def get_production_rate(self, downstream_stage: str) -> float:
        """Rate multiplier for upstream (1.0=full speed, 0.0=stop)."""
        return max(0.0, 1.0 - self.get_pressure(downstream_stage))

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "queues": dict(self._queues),
                "pressure": {k: round(v, 2) for k, v in self._pressure.items()},
            }


# ============================================================================
# 3. Bulkhead Isolator
# ============================================================================

class BulkheadIsolator:
    """
    Failure isolation between pipeline partitions.

    From the Titanic: bulkheads prevent water from flooding the entire
    ship. Similarly, isolate pipeline sections so a failure in one
    doesn't cascade to others.

    Each partition gets its own resource budget (threads, tokens, time).
    A failure in partition A cannot consume partition B's resources.
    """

    @dataclass
    class Partition:
        name: str
        stages: List[str]
        max_threads: int = 3
        max_tokens_k: float = 10.0
        active_threads: int = 0
        tokens_used_k: float = 0.0
        failures: int = 0
        status: str = "healthy"  # healthy, degraded, isolated

    def __init__(self):
        self._partitions: Dict[str, "BulkheadIsolator.Partition"] = {}
        self._lock = threading.Lock()

    def create_partition(self, name: str, stages: List[str],
                         max_threads: int = 3, max_tokens_k: float = 10.0) -> None:
        with self._lock:
            self._partitions[name] = self.Partition(
                name=name, stages=stages,
                max_threads=max_threads, max_tokens_k=max_tokens_k,
            )

    def can_execute(self, partition: str) -> bool:
        """Check if partition has resources available."""
        with self._lock:
            p = self._partitions.get(partition)
            if not p:
                return True
            if p.status == "isolated":
                return False
            return p.active_threads < p.max_threads and p.tokens_used_k < p.max_tokens_k

    def acquire(self, partition: str) -> bool:
        """Acquire a thread slot in a partition."""
        with self._lock:
            p = self._partitions.get(partition)
            if not p or not self.can_execute(partition):
                return False
            p.active_threads += 1
            return True

    def release(self, partition: str, tokens_used_k: float = 0, failed: bool = False) -> None:
        with self._lock:
            p = self._partitions.get(partition)
            if not p:
                return
            p.active_threads = max(0, p.active_threads - 1)
            p.tokens_used_k += tokens_used_k
            if failed:
                p.failures += 1
                if p.failures >= 3:
                    p.status = "degraded"
                if p.failures >= 5:
                    p.status = "isolated"

    def reset_partition(self, partition: str) -> None:
        with self._lock:
            p = self._partitions.get(partition)
            if p:
                p.failures = 0
                p.status = "healthy"

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                name: {
                    "status": p.status, "active_threads": p.active_threads,
                    "tokens_k": round(p.tokens_used_k, 1), "failures": p.failures,
                }
                for name, p in self._partitions.items()
            }


# ============================================================================
# 4. Load Shedder
# ============================================================================

class LoadShedder:
    """
    Graceful degradation under budget pressure.

    When budget is tight, shed load by:
      Priority 1 (always run): ideation, code_generation, execution
      Priority 2 (shed first): self_reflection, debate, tree_search
      Priority 3 (shed second): paper_polish, detailed_analysis

    Shedding = skip or run in lite mode, preserving core functionality.
    """

    PRIORITY_MAP = {
        "ideation": 1,
        "experiment_design": 1,
        "code_generation": 1,
        "execution": 1,
        "analysis": 2,
        "tree_search": 2,
        "self_reflection": 3,
        "debate": 3,
        "paper_writing": 2,
        "review": 2,
    }

    def __init__(self):
        self._budget_pressure: float = 0.0  # 0=relaxed, 1=critical
        self._shed_count: int = 0
        self._lock = threading.Lock()

    def set_pressure(self, remaining_pct: float) -> None:
        """Set budget pressure from remaining budget percentage."""
        with self._lock:
            if remaining_pct > 50:
                self._budget_pressure = 0.0
            elif remaining_pct > 25:
                self._budget_pressure = 0.5
            elif remaining_pct > 10:
                self._budget_pressure = 0.8
            else:
                self._budget_pressure = 1.0

    def should_run(self, stage: str) -> str:
        """Returns "run", "lite", or "skip"."""
        with self._lock:
            priority = self.PRIORITY_MAP.get(stage, 2)
            if self._budget_pressure < 0.3:
                return "run"
            if self._budget_pressure < 0.6:
                if priority >= 3:
                    self._shed_count += 1
                    return "skip"
                return "run"
            if self._budget_pressure < 0.9:
                if priority >= 3:
                    self._shed_count += 1
                    return "skip"
                if priority >= 2:
                    return "lite"
                return "run"
            # Critical: only priority 1
            if priority >= 2:
                self._shed_count += 1
                return "skip"
            return "lite"

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "pressure": round(self._budget_pressure, 2),
                "shed_count": self._shed_count,
            }


# ============================================================================
# 5. Sliding Window Rate Limiter
# ============================================================================

class SlidingWindowRateLimiter:
    """
    Token bucket rate limiting per stage.

    Prevents any single stage from monopolizing API calls.
    Each stage gets a token bucket refilled at a fixed rate.
    Calls consume tokens; when empty, the stage must wait.
    """

    @dataclass
    class Bucket:
        tokens: float
        max_tokens: float
        refill_rate: float  # tokens per second
        last_refill: float = field(default_factory=time.time)

    def __init__(self, default_rate: float = 2.0, default_burst: float = 10.0):
        self.default_rate = default_rate
        self.default_burst = default_burst
        self._buckets: Dict[str, "SlidingWindowRateLimiter.Bucket"] = {}
        self._lock = threading.Lock()

    def _get_bucket(self, stage: str) -> "SlidingWindowRateLimiter.Bucket":
        if stage not in self._buckets:
            self._buckets[stage] = self.Bucket(
                tokens=self.default_burst,
                max_tokens=self.default_burst,
                refill_rate=self.default_rate,
            )
        return self._buckets[stage]

    def try_acquire(self, stage: str, tokens: float = 1.0) -> bool:
        """Try to consume tokens. Returns False if rate-limited."""
        with self._lock:
            bucket = self._get_bucket(stage)
            now = time.time()
            elapsed = now - bucket.last_refill
            bucket.tokens = min(bucket.max_tokens, bucket.tokens + elapsed * bucket.refill_rate)
            bucket.last_refill = now

            if bucket.tokens >= tokens:
                bucket.tokens -= tokens
                return True
            return False

    def wait_time(self, stage: str, tokens: float = 1.0) -> float:
        """How long until tokens are available?"""
        with self._lock:
            bucket = self._get_bucket(stage)
            if bucket.tokens >= tokens:
                return 0.0
            deficit = tokens - bucket.tokens
            return deficit / max(bucket.refill_rate, 0.01)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                stage: {"tokens": round(b.tokens, 1), "rate": b.refill_rate}
                for stage, b in self._buckets.items()
            }


# ============================================================================
# 6. Constant Work-In-Progress (Kanban)
# ============================================================================

class ConstantWIP:
    """
    WIP-limited Kanban scheduling for pipeline stages.

    Limits how many items are "in progress" at each stage.
    Pull-based: downstream stages pull work when ready, rather than
    upstream pushing. Prevents bottlenecks and overproduction.

    Little's Law: L = λW (avg items = arrival rate × avg wait time)
    Limiting WIP (L) reduces wait time (W) when rate (λ) is constant.
    """

    @dataclass
    class KanbanColumn:
        name: str
        wip_limit: int
        items: List[str] = field(default_factory=list)
        completed: int = 0
        blocked: int = 0

    def __init__(self, default_wip: int = 3):
        self.default_wip = default_wip
        self._columns: Dict[str, "ConstantWIP.KanbanColumn"] = {}
        self._lock = threading.Lock()

    def add_column(self, name: str, wip_limit: int = None) -> None:
        with self._lock:
            self._columns[name] = self.KanbanColumn(
                name=name, wip_limit=wip_limit or self.default_wip,
            )

    def can_pull(self, stage: str) -> bool:
        """Can this stage accept more work?"""
        with self._lock:
            col = self._columns.get(stage)
            if not col:
                return True
            return len(col.items) < col.wip_limit

    def pull(self, stage: str, item_id: str) -> bool:
        """Pull an item into a stage. Returns False if WIP-limited."""
        with self._lock:
            col = self._columns.get(stage)
            if not col:
                return True
            if len(col.items) >= col.wip_limit:
                col.blocked += 1
                return False
            col.items.append(item_id)
            return True

    def complete(self, stage: str, item_id: str) -> None:
        with self._lock:
            col = self._columns.get(stage)
            if col and item_id in col.items:
                col.items.remove(item_id)
                col.completed += 1

    def throughput(self) -> Dict[str, int]:
        with self._lock:
            return {name: col.completed for name, col in self._columns.items()}

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                name: {
                    "wip": len(col.items), "limit": col.wip_limit,
                    "completed": col.completed, "blocked": col.blocked,
                }
                for name, col in self._columns.items()
            }


# ============================================================================
# 7. Critical Path Scheduler
# ============================================================================

class CriticalPathScheduler:
    """
    CPM-based parallel stage scheduling.

    Identifies the critical path (longest dependency chain) and schedules
    non-critical stages in parallel to maximize throughput.

    Forward pass: earliest start time for each stage
    Backward pass: latest start time without delaying completion
    Float = latest_start - earliest_start (stages with 0 float are critical)
    """

    @dataclass
    class Task:
        name: str
        duration: float  # estimated seconds
        dependencies: List[str] = field(default_factory=list)
        earliest_start: float = 0.0
        latest_start: float = float('inf')
        float_time: float = 0.0

    def __init__(self):
        self._tasks: Dict[str, "CriticalPathScheduler.Task"] = {}

    def add_task(self, name: str, duration: float, dependencies: List[str] = None) -> None:
        self._tasks[name] = self.Task(name=name, duration=duration, dependencies=dependencies or [])

    def compute(self) -> Dict[str, "CriticalPathScheduler.Task"]:
        """Compute forward and backward passes."""
        # Forward pass: earliest start
        computed = set()
        for _ in range(len(self._tasks) + 1):
            for name, task in self._tasks.items():
                if name in computed:
                    continue
                deps_done = all(d in computed for d in task.dependencies)
                if deps_done:
                    if task.dependencies:
                        task.earliest_start = max(
                            self._tasks[d].earliest_start + self._tasks[d].duration
                            for d in task.dependencies
                        )
                    else:
                        task.earliest_start = 0.0
                    computed.add(name)

        # Find project end time
        end_time = max(t.earliest_start + t.duration for t in self._tasks.values()) if self._tasks else 0

        # Backward pass: latest start
        computed = set()
        # Start from tasks with no successors
        successors: Dict[str, List[str]] = defaultdict(list)
        for name, task in self._tasks.items():
            for dep in task.dependencies:
                successors[dep].append(name)

        for _ in range(len(self._tasks) + 1):
            for name, task in self._tasks.items():
                if name in computed:
                    continue
                succs = successors.get(name, [])
                if not succs:
                    task.latest_start = end_time - task.duration
                    computed.add(name)
                elif all(s in computed for s in succs):
                    task.latest_start = min(
                        self._tasks[s].latest_start for s in succs
                    ) - task.duration
                    computed.add(name)

        # Float
        for task in self._tasks.values():
            task.float_time = task.latest_start - task.earliest_start

        return dict(self._tasks)

    def get_critical_path(self) -> List[str]:
        """Get stages on the critical path (float = 0)."""
        self.compute()
        critical = [name for name, t in self._tasks.items() if abs(t.float_time) < 0.01]
        critical.sort(key=lambda n: self._tasks[n].earliest_start)
        return critical

    def get_parallel_groups(self) -> List[List[str]]:
        """Get groups of stages that can run in parallel."""
        self.compute()
        groups: Dict[float, List[str]] = defaultdict(list)
        for name, task in self._tasks.items():
            groups[round(task.earliest_start, 1)].append(name)
        return [stages for _, stages in sorted(groups.items())]

    def stats(self) -> Dict[str, Any]:
        self.compute()
        return {
            "tasks": len(self._tasks),
            "critical_path": self.get_critical_path(),
            "parallel_groups": self.get_parallel_groups(),
            "estimated_total": round(
                max((t.earliest_start + t.duration for t in self._tasks.values()), default=0), 1
            ),
        }


# ============================================================================
# 8. Loop Invariant Hoister
# ============================================================================

class LoopInvariantHoister:
    """
    Cache computations that don't change across pipeline iterations.

    Compiler optimization: if a value doesn't change inside a loop,
    compute it once before the loop and reuse.

    Applied to pipeline:
      - DAG construction: same topic → same DAG across iterations
      - System prompts: same agent role → same system prompt
      - Config values: don't re-read config every call
    """

    def __init__(self):
        self._invariants: Dict[str, Any] = {}
        self._computation_count: Dict[str, int] = defaultdict(int)
        self._hoist_savings: int = 0
        self._lock = threading.Lock()

    def compute_once(self, key: str, compute_fn: Callable[[], Any]) -> Any:
        """Compute a value once and cache it (loop-invariant hoisting)."""
        with self._lock:
            if key in self._invariants:
                self._hoist_savings += 1
                return self._invariants[key]

        value = compute_fn()
        with self._lock:
            self._invariants[key] = value
            self._computation_count[key] += 1
        return value

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._invariants.pop(key, None)

    def invalidate_all(self) -> None:
        with self._lock:
            self._invariants.clear()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "cached_invariants": len(self._invariants),
                "hoist_savings": self._hoist_savings,
                "computations": dict(self._computation_count),
            }


# ============================================================================
# 9. Betweenness Centrality (Bottleneck Detection)
# ============================================================================

class BetweennessCentrality:
    """
    Network bottleneck detection in the pipeline stage graph.

    Betweenness centrality measures how often a node lies on shortest
    paths between other nodes. High-betweenness stages are bottlenecks:
    if they fail or slow down, many other stages are affected.

    Used to:
      - Identify which stages need the most reliability investment
      - Prioritize caching/optimization for high-centrality stages
      - Detect single points of failure
    """

    def __init__(self):
        self._edges: List[Tuple[str, str]] = []
        self._nodes: Set[str] = set()

    def add_edge(self, src: str, dst: str) -> None:
        self._edges.append((src, dst))
        self._nodes.add(src)
        self._nodes.add(dst)

    def compute(self) -> Dict[str, float]:
        """Compute betweenness centrality for all nodes."""
        if not self._nodes:
            return {}

        centrality = {n: 0.0 for n in self._nodes}
        nodes = sorted(self._nodes)

        # Build adjacency
        adj: Dict[str, List[str]] = defaultdict(list)
        for src, dst in self._edges:
            adj[src].append(dst)

        # BFS from each node
        for source in nodes:
            # BFS
            dist: Dict[str, int] = {source: 0}
            paths: Dict[str, int] = {source: 1}
            queue = deque([source])
            order = []

            while queue:
                v = queue.popleft()
                order.append(v)
                for w in adj.get(v, []):
                    if w not in dist:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    if dist[w] == dist[v] + 1:
                        paths[w] = paths.get(w, 0) + paths[v]

            # Accumulate
            delta = {n: 0.0 for n in nodes}
            for w in reversed(order):
                for v in adj.get(w, []):
                    if dist.get(v, -1) == dist.get(w, -1) + 1:
                        ratio = paths.get(w, 0) / max(paths.get(v, 1), 1)
                        delta[w] += ratio * (1 + delta[v])
                if w != source:
                    centrality[w] += delta[w]

        # Normalize
        n = len(nodes)
        if n > 2:
            norm = 1.0 / ((n - 1) * (n - 2))
            centrality = {k: v * norm for k, v in centrality.items()}

        return centrality

    def get_bottlenecks(self, top_n: int = 3) -> List[Tuple[str, float]]:
        bc = self.compute()
        return sorted(bc.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def stats(self) -> Dict[str, Any]:
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "bottlenecks": self.get_bottlenecks(3),
        }


# ============================================================================
# 10. PID Controller
# ============================================================================

class PIDController:
    """
    Closed-loop feedback control for quality targeting.

    PID (Proportional-Integral-Derivative) controller adjusts a control
    variable (e.g., temperature, budget allocation) to minimize error
    between actual and target quality.

    u(t) = Kp×e(t) + Ki×∫e(τ)dτ + Kd×de/dt

    Used to automatically tune pipeline parameters toward a quality target.
    """

    def __init__(self, kp: float = 0.5, ki: float = 0.1, kd: float = 0.05,
                 target: float = 0.7, output_min: float = 0.1, output_max: float = 0.95):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.output_min = output_min
        self.output_max = output_max
        self._integral = 0.0
        self._prev_error = 0.0
        self._history: List[Tuple[float, float, float]] = []  # (error, output, timestamp)

    def update(self, measured: float, dt: float = 1.0) -> float:
        """Compute control output given current measurement."""
        error = self.target - measured

        self._integral += error * dt
        # Anti-windup: clamp integral
        self._integral = max(-2.0, min(2.0, self._integral))

        derivative = (error - self._prev_error) / max(dt, 0.001)
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        output = max(self.output_min, min(self.output_max, 0.5 + output))

        self._history.append((error, output, time.time()))
        return output

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0

    @property
    def is_converged(self) -> bool:
        if len(self._history) < 3:
            return False
        recent_errors = [abs(e) for e, _, _ in self._history[-3:]]
        return max(recent_errors) < 0.05

    def stats(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "converged": self.is_converged,
            "integral": round(self._integral, 3),
            "last_error": round(self._prev_error, 3),
            "history_len": len(self._history),
            "last_output": round(self._history[-1][1], 3) if self._history else None,
        }


# ============================================================================
# 11. Exponential Backoff Pool
# ============================================================================

class ExponentialBackoffPool:
    """
    Coordinated exponential backoff across all agents.

    When multiple agents hit rate limits simultaneously, they need
    coordinated backoff to prevent thundering herd. This pool
    maintains a global backoff state that all agents share.

    Backoff doubles on each consecutive failure, resets on success.
    Jitter prevents synchronized retries.
    """

    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0, jitter: float = 0.5):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self._consecutive_failures: int = 0
        self._current_delay: float = 0.0
        self._total_wait: float = 0.0
        self._lock = threading.Lock()

    def record_failure(self) -> float:
        """Record a failure. Returns recommended wait time in seconds."""
        with self._lock:
            self._consecutive_failures += 1
            base = min(self.base_delay * (2 ** self._consecutive_failures), self.max_delay)
            jitter_amount = base * self.jitter * (random.random() * 2 - 1)
            self._current_delay = max(0.1, base + jitter_amount)
            self._total_wait += self._current_delay
            return self._current_delay

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._current_delay = 0.0

    @property
    def should_wait(self) -> bool:
        with self._lock:
            return self._consecutive_failures > 0

    @property
    def current_delay(self) -> float:
        with self._lock:
            return self._current_delay

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "consecutive_failures": self._consecutive_failures,
                "current_delay": round(self._current_delay, 2),
                "total_wait": round(self._total_wait, 2),
            }


# ============================================================================
# 12. Checkpoint Recovery
# ============================================================================

class CheckpointRecovery:
    """
    Transactional stage execution with rollback capability.

    Before each stage, save a checkpoint. If the stage fails, roll back
    to the checkpoint and retry (or skip). This provides ACID-like
    guarantees for the pipeline.

    Checkpoint = snapshot of all relevant state at a point in time.
    """

    @dataclass
    class Checkpoint:
        stage: str
        state: Dict[str, Any]
        timestamp: float = field(default_factory=time.time)
        valid: bool = True

    def __init__(self, max_checkpoints: int = 20):
        self.max_checkpoints = max_checkpoints
        self._checkpoints: Dict[str, "CheckpointRecovery.Checkpoint"] = {}
        self._rollback_count: int = 0
        self._lock = threading.Lock()

    def save(self, stage: str, state: Dict[str, Any]) -> None:
        """Save a checkpoint before stage execution."""
        with self._lock:
            self._checkpoints[stage] = self.Checkpoint(
                stage=stage, state=dict(state),
            )
            # Evict old checkpoints
            if len(self._checkpoints) > self.max_checkpoints:
                oldest = min(self._checkpoints, key=lambda k: self._checkpoints[k].timestamp)
                del self._checkpoints[oldest]

    def load(self, stage: str) -> Optional[Dict[str, Any]]:
        """Load a checkpoint for rollback."""
        with self._lock:
            cp = self._checkpoints.get(stage)
            if cp and cp.valid:
                self._rollback_count += 1
                return dict(cp.state)
            return None

    def invalidate(self, stage: str) -> None:
        """Mark a checkpoint as consumed (stage succeeded)."""
        with self._lock:
            cp = self._checkpoints.get(stage)
            if cp:
                cp.valid = False

    def get_latest_valid(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Get the most recent valid checkpoint for recovery."""
        with self._lock:
            valid = [(k, cp) for k, cp in self._checkpoints.items() if cp.valid]
            if not valid:
                return None
            latest = max(valid, key=lambda x: x[1].timestamp)
            return latest[0], dict(latest[1].state)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            valid = sum(1 for cp in self._checkpoints.values() if cp.valid)
            return {
                "checkpoints": len(self._checkpoints),
                "valid": valid,
                "rollbacks": self._rollback_count,
            }


# ============================================================================
# Master Systems Optimizer
# ============================================================================

class SystemsOptimizer:
    """Aggregates all systems/reliability optimization techniques."""

    def __init__(self, enable_all: bool = True):
        self.raft = RaftConsensus() if enable_all else None
        self.backpressure = BackpressureController() if enable_all else None
        self.bulkhead = BulkheadIsolator() if enable_all else None
        self.load_shedder = LoadShedder() if enable_all else None
        self.rate_limiter = SlidingWindowRateLimiter() if enable_all else None
        self.kanban = ConstantWIP() if enable_all else None
        self.cpm = CriticalPathScheduler() if enable_all else None
        self.hoister = LoopInvariantHoister() if enable_all else None
        self.centrality = BetweennessCentrality() if enable_all else None
        self.pid = PIDController() if enable_all else None
        self.backoff_pool = ExponentialBackoffPool() if enable_all else None
        self.checkpoint = CheckpointRecovery() if enable_all else None

    def summary(self) -> Dict[str, Any]:
        result = {}
        if self.raft: result["raft_consensus"] = self.raft.stats()
        if self.backpressure: result["backpressure"] = self.backpressure.stats()
        if self.bulkhead: result["bulkhead"] = self.bulkhead.stats()
        if self.load_shedder: result["load_shedder"] = self.load_shedder.stats()
        if self.rate_limiter: result["rate_limiter"] = self.rate_limiter.stats()
        if self.kanban: result["kanban"] = self.kanban.stats()
        if self.cpm: result["critical_path"] = self.cpm.stats()
        if self.hoister: result["loop_hoister"] = self.hoister.stats()
        if self.centrality: result["centrality"] = self.centrality.stats()
        if self.pid: result["pid_controller"] = self.pid.stats()
        if self.backoff_pool: result["backoff_pool"] = self.backoff_pool.stats()
        if self.checkpoint: result["checkpoint"] = self.checkpoint.stats()
        return result
