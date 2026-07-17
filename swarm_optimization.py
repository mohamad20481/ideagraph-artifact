"""
swarm_optimization.py - Agent swarm coordination for IdeaGraph.

Layer 10: True multi-agent collaboration patterns — spawning, communication,
consensus, delegation, negotiation, and emergent self-organization. These
transform the pipeline from a sequential singleton chain into a dynamic
swarm of cooperating agents.

  1.  AgentPool                — Worker pool with lifecycle management
  2.  MessageBus               — Pub/sub event system for inter-agent comms
  3.  SharedBlackboard         — Collaborative KV workspace with subscriptions
  4.  SwarmConsensus           — N-way consensus (vote, average, debate, Delphi)
  5.  HierarchicalCoordinator  — Manager-worker delegation with rollup
  6.  SpecialistRouter         — Route tasks by capability matching
  7.  DynamicTeamFormer        — Assemble optimal agent teams per task
  8.  StigmergyEngine          — Indirect coordination via digital pheromones
  9.  AgentNegotiator          — Contract-net protocol for task allocation
  10. SwarmMemoryPool          — Collective short/long-term shared memory
  11. EmergentBehaviorDetector  — Detect and amplify emergent swarm patterns
  12. MultiAgentDebateArena    — N-way structured debate with moderator
"""

from __future__ import annotations

import fnmatch
import math
import random
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ============================================================================
# 1. Agent Pool
# ============================================================================

@dataclass
class SwarmAgentHandle:
    """Handle representing a managed agent in the pool."""
    agent_id: str
    role: str
    capabilities: List[str] = field(default_factory=list)
    status: str = "active"  # active, busy, retired
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_duration: float = 0.0


class AgentPool:
    """
    Worker pool managing agent lifecycle — spawn, retire, health tracking.

    Each agent gets a SwarmAgentHandle with unique ID, role, and capabilities.
    Tasks are submitted to a shared ThreadPoolExecutor.
    Idle agents are retired after timeout. Unhealthy agents are replaced.
    """

    def __init__(self, max_agents: int = 12, idle_timeout_s: float = 120.0):
        self.max_agents = max_agents
        self.idle_timeout = idle_timeout_s
        self._agents: Dict[str, SwarmAgentHandle] = {}
        self._executor: Optional[ThreadPoolExecutor] = None  # lazy init
        self._lock = threading.Lock()

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_agents)
        return self._executor

    def spawn(self, role: str, capabilities: List[str] = None) -> SwarmAgentHandle:
        """Spawn a new agent in the pool."""
        with self._lock:
            if len(self._agents) >= self.max_agents:
                self._retire_oldest()
            agent_id = f"{role}_{uuid.uuid4().hex[:6]}"
            handle = SwarmAgentHandle(
                agent_id=agent_id, role=role,
                capabilities=capabilities or [],
            )
            self._agents[agent_id] = handle
            return handle

    def retire(self, agent_id: str) -> None:
        with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id].status = "retired"

    def _retire_oldest(self) -> None:
        """Retire the least recently active agent."""
        active = {k: v for k, v in self._agents.items() if v.status == "active"}
        if active:
            oldest_id = min(active, key=lambda k: active[k].last_active)
            self._agents[oldest_id].status = "retired"

    def get_agent(self, agent_id: str) -> Optional[SwarmAgentHandle]:
        return self._agents.get(agent_id)

    def get_active(self, role: str = None) -> List[SwarmAgentHandle]:
        with self._lock:
            agents = [a for a in self._agents.values() if a.status == "active"]
            if role:
                agents = [a for a in agents if a.role == role]
            return agents

    def submit_task(self, agent_id: str, fn: Callable, *args, **kwargs) -> Optional[Future]:
        """Submit a task for an agent to execute."""
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent or agent.status == "retired":
                return None
            agent.status = "busy"
            agent.last_active = time.time()

        def _wrapped():
            start = time.time()
            try:
                result = fn(*args, **kwargs)
                with self._lock:
                    agent.tasks_completed += 1
                    agent.total_duration += time.time() - start
                    agent.status = "active"
                return result
            except Exception as e:
                with self._lock:
                    agent.tasks_failed += 1
                    agent.status = "active"
                raise

        return self._get_executor().submit(_wrapped)

    def health_check(self) -> List[str]:
        """Retire idle agents. Returns list of retired agent IDs."""
        retired = []
        now = time.time()
        with self._lock:
            for agent_id, agent in self._agents.items():
                if agent.status == "active" and (now - agent.last_active) > self.idle_timeout:
                    agent.status = "retired"
                    retired.append(agent_id)
        return retired

    def shutdown(self) -> None:
        if self._executor:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                self._executor.shutdown(wait=False)
            self._executor = None

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            active = sum(1 for a in self._agents.values() if a.status == "active")
            busy = sum(1 for a in self._agents.values() if a.status == "busy")
            retired = sum(1 for a in self._agents.values() if a.status == "retired")
            total_tasks = sum(a.tasks_completed for a in self._agents.values())
        return {
            "total_agents": len(self._agents),
            "active": active, "busy": busy, "retired": retired,
            "total_tasks": total_tasks,
            "roles": list(set(a.role for a in self._agents.values() if a.status != "retired")),
        }


# ============================================================================
# 2. Message Bus
# ============================================================================

@dataclass
class SwarmMessage:
    """A message on the bus."""
    topic: str
    payload: Any
    sender_id: str
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


class MessageBus:
    """
    In-process pub/sub event system for inter-agent communication.

    Agents publish messages to topics; subscribers receive callbacks.
    History buffer enables replay for late-joining agents.
    """

    def __init__(self, history_size: int = 500):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._history: deque = deque(maxlen=history_size)
        self._message_count = 0
        self._lock = threading.Lock()

    def subscribe(self, topic: str, callback: Callable[[SwarmMessage], None]) -> None:
        with self._lock:
            self._subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable) -> None:
        with self._lock:
            subs = self._subscribers.get(topic, [])
            if callback in subs:
                subs.remove(callback)

    def publish(self, topic: str, payload: Any, sender_id: str = "") -> SwarmMessage:
        """Publish a message to all subscribers of a topic."""
        msg = SwarmMessage(topic=topic, payload=payload, sender_id=sender_id)
        with self._lock:
            self._history.append(msg)
            self._message_count += 1
            callbacks = list(self._subscribers.get(topic, []))
        for cb in callbacks:
            try:
                cb(msg)
            except Exception:
                pass
        return msg

    def get_history(self, topic: str = None, n: int = 20) -> List[SwarmMessage]:
        with self._lock:
            msgs = list(self._history)
        if topic:
            msgs = [m for m in msgs if m.topic == topic]
        return msgs[-n:]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "topics": len(self._subscribers),
                "total_subscribers": sum(len(v) for v in self._subscribers.values()),
                "messages_published": self._message_count,
                "history_size": len(self._history),
            }


# ============================================================================
# 3. Shared Blackboard
# ============================================================================

@dataclass
class BlackboardEntry:
    value: Any
    writer_id: str
    timestamp: float = field(default_factory=time.time)
    version: int = 1


class SharedBlackboard:
    """
    Collaborative key-value workspace with versioning and subscriptions.

    Agents read/write to a shared workspace. Changes trigger subscriber
    notifications. Version tracking enables conflict detection.
    """

    def __init__(self):
        self._entries: Dict[str, BlackboardEntry] = {}
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._read_count = 0
        self._write_count = 0
        self._lock = threading.RLock()  # reentrant for nested reads

    def write(self, key: str, value: Any, writer_id: str = "") -> int:
        """Write a value. Returns new version number."""
        with self._lock:
            existing = self._entries.get(key)
            version = (existing.version + 1) if existing else 1
            self._entries[key] = BlackboardEntry(
                value=value, writer_id=writer_id, version=version,
            )
            self._write_count += 1
            callbacks = list(self._subscribers.get(key, []))
        for cb in callbacks:
            try:
                cb(key, value, writer_id)
            except Exception:
                pass
        return version

    def read(self, key: str) -> Optional[Any]:
        with self._lock:
            self._read_count += 1
            entry = self._entries.get(key)
            return entry.value if entry else None

    def read_pattern(self, pattern: str) -> Dict[str, Any]:
        """Read all keys matching a glob pattern."""
        with self._lock:
            return {
                k: e.value for k, e in self._entries.items()
                if fnmatch.fnmatch(k, pattern)
            }

    def subscribe(self, key: str, callback: Callable) -> None:
        with self._lock:
            self._subscribers[key].append(callback)

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._entries.keys())

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "reads": self._read_count,
                "writes": self._write_count,
                "subscribers": sum(len(v) for v in self._subscribers.values()),
                "writers": len(set(e.writer_id for e in self._entries.values())),
            }


# ============================================================================
# 4. Swarm Consensus
# ============================================================================

class ConsensusStrategy(Enum):
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    DEBATE = "debate"
    DELPHI = "delphi"


class SwarmConsensus:
    """
    N-way consensus via multiple strategies.

    - MAJORITY_VOTE: count occurrences, return most common (categorical)
    - WEIGHTED_AVERAGE: precision-weighted mean (numeric)
    - DEBATE: structured argumentation with judge (uses LLM)
    - DELPHI: multi-round anonymous aggregation → convergence
    """

    def __init__(self):
        self._history: List[Dict] = []
        self._lock = threading.Lock()

    def reach_consensus(
        self, opinions: Dict[str, Any],
        strategy: ConsensusStrategy = ConsensusStrategy.WEIGHTED_AVERAGE,
        weights: Dict[str, float] = None,
    ) -> Any:
        """Aggregate opinions into consensus."""
        if not opinions:
            return None

        weights = weights or {k: 1.0 for k in opinions}

        if strategy == ConsensusStrategy.MAJORITY_VOTE:
            result = self._majority_vote(opinions)
        elif strategy == ConsensusStrategy.WEIGHTED_AVERAGE:
            result = self._weighted_average(opinions, weights)
        elif strategy == ConsensusStrategy.DELPHI:
            result = self._delphi(opinions, weights)
        else:
            result = self._weighted_average(opinions, weights)

        with self._lock:
            self._history.append({
                "strategy": strategy.value,
                "n_opinions": len(opinions),
                "result": result,
                "timestamp": time.time(),
            })
        return result

    def _majority_vote(self, opinions: Dict[str, Any]) -> Any:
        counts: Dict[Any, int] = defaultdict(int)
        for value in opinions.values():
            # Handle dicts by converting to string key
            key = str(value) if isinstance(value, dict) else value
            counts[key] = counts.get(key, 0) + 1
        return max(counts, key=counts.get)

    def _weighted_average(self, opinions: Dict[str, Any], weights: Dict[str, float]) -> float:
        total_w = 0
        weighted_sum = 0
        for agent_id, value in opinions.items():
            if isinstance(value, (int, float)):
                w = weights.get(agent_id, 1.0)
                weighted_sum += w * value
                total_w += w
        return weighted_sum / max(total_w, 0.001)

    def _delphi(self, opinions: Dict[str, Any], weights: Dict[str, float], rounds: int = 3) -> float:
        """Multi-round Delphi method: share anonymous summary, re-estimate, converge."""
        current = dict(opinions)
        for _ in range(rounds):
            # Compute anonymous summary
            values = [v for v in current.values() if isinstance(v, (int, float))]
            if not values:
                break
            median = sorted(values)[len(values) // 2]
            mean = sum(values) / len(values)

            # Each agent adjusts toward the group (30% adjustment)
            new_opinions = {}
            for agent_id, value in current.items():
                if isinstance(value, (int, float)):
                    new_opinions[agent_id] = value + 0.3 * (median - value)
                else:
                    new_opinions[agent_id] = value
            current = new_opinions

        return self._weighted_average(current, weights)

    def agreement_rate(self, opinions: Dict[str, Any], threshold: float = 0.1) -> float:
        """Fraction of agents within threshold of consensus."""
        values = [v for v in opinions.values() if isinstance(v, (int, float))]
        if len(values) < 2:
            return 1.0
        mean = sum(values) / len(values)
        agreeing = sum(1 for v in values if abs(v - mean) <= threshold)
        return agreeing / len(values)

    def stats(self) -> Dict[str, Any]:
        by_strategy = defaultdict(int)
        for h in self._history:
            by_strategy[h["strategy"]] += 1
        return {
            "total_consensus": len(self._history),
            "by_strategy": dict(by_strategy),
        }


# ============================================================================
# 5. Hierarchical Coordinator
# ============================================================================

@dataclass
class CoordinatorNode:
    agent_id: str
    role: str
    children: List["CoordinatorNode"] = field(default_factory=list)
    parent_id: Optional[str] = None
    depth: int = 0
    result: Any = None
    status: str = "pending"


class HierarchicalCoordinator:
    """
    Manager-worker delegation chains with result rollup.

    Builds a tree of agents. Manager delegates subtasks to children,
    collects results, and merges them bottom-up.
    """

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self._root: Optional[CoordinatorNode] = None
        self._nodes: Dict[str, CoordinatorNode] = {}
        self._delegation_count = 0
        self._lock = threading.Lock()

    def create_hierarchy(self, structure: Dict[str, Any]) -> CoordinatorNode:
        """
        Build hierarchy from nested dict.
        Example: {"manager": {"role": "lead", "children": [{"role": "coder"}, {"role": "reviewer"}]}}
        """
        def _build(spec: Dict, parent_id: str = None, depth: int = 0) -> CoordinatorNode:
            agent_id = f"{spec.get('role', 'agent')}_{uuid.uuid4().hex[:4]}"
            node = CoordinatorNode(
                agent_id=agent_id,
                role=spec.get("role", "worker"),
                parent_id=parent_id,
                depth=depth,
            )
            self._nodes[agent_id] = node
            if depth < self.max_depth:
                for child_spec in spec.get("children", []):
                    child = _build(child_spec, agent_id, depth + 1)
                    node.children.append(child)
            return node

        self._root = _build(structure)
        return self._root

    def delegate(self, node: CoordinatorNode, task: Any,
                 executor_fn: Callable[[str, Any], Any]) -> Any:
        """Recursively delegate task to children and rollup results."""
        with self._lock:
            self._delegation_count += 1

        if not node.children:
            # Leaf node: execute directly
            node.result = executor_fn(node.agent_id, task)
            node.status = "completed"
            return node.result

        # Delegate to children
        child_results = []
        for child in node.children:
            result = self.delegate(child, task, executor_fn)
            child_results.append(result)

        # Rollup: manager merges child results
        node.result = self._merge_results(child_results)
        node.status = "completed"
        return node.result

    def _merge_results(self, results: List[Any]) -> Any:
        """Merge results from children (simple: collect non-None)."""
        merged = [r for r in results if r is not None]
        if not merged:
            return None
        if all(isinstance(r, (int, float)) for r in merged):
            return sum(merged) / len(merged)
        return merged

    def stats(self) -> Dict[str, Any]:
        return {
            "nodes": len(self._nodes),
            "max_depth": self.max_depth,
            "delegations": self._delegation_count,
            "root": self._root.role if self._root else None,
        }


# ============================================================================
# 6. Specialist Router
# ============================================================================

class SpecialistRouter:
    """
    Route tasks to agents by capability matching.

    Match score = Jaccard similarity between required and agent capabilities.
    Tiebreak by load (fewer active tasks preferred).
    """

    def __init__(self):
        self._registry: Dict[str, Set[str]] = {}  # agent_id → capabilities
        self._load: Dict[str, int] = defaultdict(int)
        self._routing_count = 0
        self._lock = threading.Lock()

    def register(self, agent_id: str, capabilities: List[str]) -> None:
        with self._lock:
            self._registry[agent_id] = set(c.lower() for c in capabilities)

    def unregister(self, agent_id: str) -> None:
        with self._lock:
            self._registry.pop(agent_id, None)

    def route(self, requirements: List[str], n: int = 1) -> List[Tuple[str, float]]:
        """Find best-matching agents. Returns [(agent_id, match_score)]."""
        with self._lock:
            self._routing_count += 1
            req_set = set(r.lower() for r in requirements)
            scored = []
            for agent_id, caps in self._registry.items():
                if not req_set:
                    scored.append((agent_id, 0.5))
                    continue
                intersection = len(req_set & caps)
                union = len(req_set | caps)
                jaccard = intersection / max(union, 1)
                # Penalize high load
                load_penalty = 1.0 / (1 + self._load.get(agent_id, 0) * 0.1)
                scored.append((agent_id, jaccard * load_penalty))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def record_task_start(self, agent_id: str) -> None:
        with self._lock:
            self._load[agent_id] += 1

    def record_task_end(self, agent_id: str) -> None:
        with self._lock:
            self._load[agent_id] = max(0, self._load.get(agent_id, 0) - 1)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "registered_agents": len(self._registry),
                "routing_calls": self._routing_count,
                "capability_distribution": {
                    agent_id: sorted(caps) for agent_id, caps in list(self._registry.items())[:5]
                },
            }


# ============================================================================
# 7. Dynamic Team Former
# ============================================================================

@dataclass
class TeamAssignment:
    task_id: str
    agents: List[SwarmAgentHandle]
    formed_at: float = field(default_factory=time.time)
    disbanded_at: float = 0.0
    quality: float = 0.0


class DynamicTeamFormer:
    """
    Assemble optimal agent teams per task with diversity constraint.

    Uses SpecialistRouter for capability matching + ensures no two
    team members have identical capability sets (diversity).
    """

    def __init__(self, pool: AgentPool = None, router: SpecialistRouter = None):
        self._pool = pool
        self._router = router
        self._teams: Dict[str, TeamAssignment] = {}
        self._lock = threading.Lock()

    def form_team(
        self, task_id: str, required_roles: List[str],
        team_size: int = 3,
    ) -> TeamAssignment:
        """Assemble a team for a task."""
        agents = []
        seen_cap_sets = set()

        if self._pool:
            for role in required_roles[:team_size]:
                available = self._pool.get_active(role=role)
                for agent in available:
                    cap_key = frozenset(agent.capabilities)
                    if cap_key not in seen_cap_sets:
                        agents.append(agent)
                        seen_cap_sets.add(cap_key)
                        break

            # Fill remaining slots from any active agents
            while len(agents) < team_size:
                all_active = self._pool.get_active()
                for agent in all_active:
                    if agent not in agents:
                        cap_key = frozenset(agent.capabilities)
                        if cap_key not in seen_cap_sets:
                            agents.append(agent)
                            seen_cap_sets.add(cap_key)
                            break
                else:
                    break

        team = TeamAssignment(task_id=task_id, agents=agents)
        with self._lock:
            self._teams[task_id] = team
        return team

    def disband_team(self, task_id: str, quality: float = 0.0) -> None:
        with self._lock:
            team = self._teams.get(task_id)
            if team:
                team.disbanded_at = time.time()
                team.quality = quality

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            active = sum(1 for t in self._teams.values() if t.disbanded_at == 0)
            disbanded = sum(1 for t in self._teams.values() if t.disbanded_at > 0)
        return {
            "teams_formed": len(self._teams),
            "active_teams": active,
            "disbanded_teams": disbanded,
        }


# ============================================================================
# 8. Stigmergy Engine
# ============================================================================

@dataclass
class StigmergyMarker:
    value: Any
    strength: float
    depositor_id: str
    timestamp: float = field(default_factory=time.time)


class StigmergyEngine:
    """
    Indirect coordination via digital pheromone markers.

    Agents deposit markers in a shared environment. Other agents sense
    markers to guide their behavior. Markers decay over time (evaporation)
    so stale information fades naturally.

    Like ants leaving pheromone trails — no direct communication needed.
    """

    def __init__(self, evaporation_rate: float = 0.05):
        self.evap_rate = evaporation_rate
        self._markers: Dict[str, StigmergyMarker] = {}
        self._deposit_count = 0
        self._sense_count = 0
        self._lock = threading.RLock()  # reentrant for nested calls

    def deposit(self, key: str, value: Any, strength: float = 1.0,
                depositor_id: str = "") -> None:
        """Deposit or strengthen a pheromone marker."""
        with self._lock:
            existing = self._markers.get(key)
            if existing:
                existing.strength += strength
                existing.value = value
                existing.timestamp = time.time()
            else:
                self._markers[key] = StigmergyMarker(
                    value=value, strength=strength, depositor_id=depositor_id,
                )
            self._deposit_count += 1

    def sense(self, key: str) -> Optional[Tuple[Any, float]]:
        """Read a marker's value and time-decayed strength."""
        with self._lock:
            self._sense_count += 1
            marker = self._markers.get(key)
            if not marker:
                return None
            age = time.time() - marker.timestamp
            decayed_strength = marker.strength * math.exp(-self.evap_rate * age)
            if decayed_strength < 0.01:
                del self._markers[key]
                return None
            return marker.value, decayed_strength

    def sense_pattern(self, pattern: str) -> Dict[str, Tuple[Any, float]]:
        """Sense all markers matching a glob pattern."""
        with self._lock:
            results = {}
            now = time.time()
            for key, marker in list(self._markers.items()):
                if fnmatch.fnmatch(key, pattern):
                    age = now - marker.timestamp
                    strength = marker.strength * math.exp(-self.evap_rate * age)
                    if strength >= 0.01:
                        results[key] = (marker.value, strength)
            return results

    def evaporate(self) -> int:
        """Explicit evaporation pass. Returns number of markers removed."""
        removed = 0
        now = time.time()
        with self._lock:
            to_remove = []
            for key, marker in self._markers.items():
                age = now - marker.timestamp
                marker.strength *= math.exp(-self.evap_rate * min(age, 10))
                if marker.strength < 0.01:
                    to_remove.append(key)
            for key in to_remove:
                del self._markers[key]
                removed += 1
        return removed

    def strongest(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get the n strongest markers."""
        with self._lock:
            now = time.time()
            scored = []
            for key, marker in self._markers.items():
                age = now - marker.timestamp
                strength = marker.strength * math.exp(-self.evap_rate * age)
                scored.append((key, strength))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "active_markers": len(self._markers),
                "deposits": self._deposit_count,
                "senses": self._sense_count,
                "strongest": self.strongest(3),
            }


# ============================================================================
# 9. Agent Negotiator (Contract Net)
# ============================================================================

@dataclass
class ContractBid:
    task_id: str
    agent_id: str
    cost_estimate: float
    time_estimate: float
    confidence: float
    timestamp: float = field(default_factory=time.time)


class AgentNegotiator:
    """
    Contract-net protocol for task allocation.

    1. Manager announces task with requirements
    2. Agents submit bids (cost, time, confidence)
    3. Manager awards to best bid (highest confidence/cost ratio)
    4. Winner executes; losers stand down
    """

    def __init__(self):
        self._auctions: Dict[str, Dict] = {}  # task_id → auction state
        self._bids: Dict[str, List[ContractBid]] = defaultdict(list)
        self._awards: Dict[str, str] = {}  # task_id → winning agent_id
        self._lock = threading.Lock()

    def announce(self, task_id: str, requirements: List[str],
                 budget_limit: float = float("inf")) -> None:
        """Manager announces a task for bidding."""
        with self._lock:
            self._auctions[task_id] = {
                "requirements": requirements,
                "budget_limit": budget_limit,
                "announced_at": time.time(),
                "status": "open",
            }

    def bid(self, task_id: str, agent_id: str, cost: float,
            time_est: float, confidence: float) -> bool:
        """Agent submits a bid. Returns True if bid accepted."""
        with self._lock:
            auction = self._auctions.get(task_id)
            if not auction or auction["status"] != "open":
                return False
            if cost > auction["budget_limit"]:
                return False
            self._bids[task_id].append(ContractBid(
                task_id=task_id, agent_id=agent_id,
                cost_estimate=cost, time_estimate=time_est,
                confidence=confidence,
            ))
            return True

    def award(self, task_id: str) -> Optional[str]:
        """Award task to best bidder. Returns winning agent_id."""
        with self._lock:
            bids = self._bids.get(task_id, [])
            if not bids:
                return None
            # Score: confidence / (cost + 0.01) — high confidence, low cost wins
            best = max(bids, key=lambda b: b.confidence / max(b.cost_estimate, 0.01))
            self._awards[task_id] = best.agent_id
            self._auctions[task_id]["status"] = "awarded"
            return best.agent_id

    def get_winner(self, task_id: str) -> Optional[str]:
        return self._awards.get(task_id)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "auctions": len(self._auctions),
                "total_bids": sum(len(b) for b in self._bids.values()),
                "awarded": len(self._awards),
                "avg_bids_per_auction": round(
                    sum(len(b) for b in self._bids.values()) / max(len(self._auctions), 1), 1
                ),
            }


# ============================================================================
# 10. Swarm Memory Pool
# ============================================================================

class SwarmMemoryPool:
    """
    Collective short-term + long-term shared memory with consolidation.

    Short-term: recent, high-churn items (decaying)
    Long-term: consolidated, stable items (persistent)

    Consolidation: frequently-accessed short-term items promote to long-term.
    Forgetting: old, rarely-accessed items are removed.
    """

    @dataclass
    class MemoryItem:
        key: str
        value: Any
        agent_id: str
        access_count: int = 0
        created_at: float = field(default_factory=time.time)
        last_accessed: float = field(default_factory=time.time)

    def __init__(self, stm_capacity: int = 200, ltm_capacity: int = 500):
        self._stm: Dict[str, "SwarmMemoryPool.MemoryItem"] = {}
        self._ltm: Dict[str, "SwarmMemoryPool.MemoryItem"] = {}
        self._stm_capacity = stm_capacity
        self._ltm_capacity = ltm_capacity
        self._lock = threading.Lock()

    def store(self, key: str, value: Any, agent_id: str = "") -> None:
        """Store in short-term memory."""
        with self._lock:
            if len(self._stm) >= self._stm_capacity:
                # Evict least accessed
                evict_key = min(self._stm, key=lambda k: self._stm[k].access_count)
                del self._stm[evict_key]
            self._stm[key] = self.MemoryItem(key=key, value=value, agent_id=agent_id)

    def recall(self, key: str) -> Optional[Any]:
        """Recall from STM first, then LTM."""
        with self._lock:
            item = self._stm.get(key) or self._ltm.get(key)
            if item:
                item.access_count += 1
                item.last_accessed = time.time()
                return item.value
            return None

    def recall_pattern(self, pattern: str, n: int = 5) -> List[Tuple[str, Any]]:
        """Recall items matching a pattern, sorted by access count."""
        with self._lock:
            all_items = {**self._stm, **self._ltm}
            matches = [
                (k, item) for k, item in all_items.items()
                if fnmatch.fnmatch(k, pattern)
            ]
        matches.sort(key=lambda x: x[1].access_count, reverse=True)
        return [(k, item.value) for k, item in matches[:n]]

    def consolidate(self, min_access: int = 3) -> int:
        """Promote frequently-accessed STM items to LTM."""
        promoted = 0
        with self._lock:
            to_promote = [
                k for k, item in self._stm.items()
                if item.access_count >= min_access
            ]
            for key in to_promote:
                if len(self._ltm) >= self._ltm_capacity:
                    evict_key = min(self._ltm, key=lambda k: self._ltm[k].access_count)
                    del self._ltm[evict_key]
                self._ltm[key] = self._stm.pop(key)
                promoted += 1
        return promoted

    def forget(self, max_age_s: float = 3600) -> int:
        """Remove old, rarely-accessed items."""
        removed = 0
        now = time.time()
        with self._lock:
            for store in [self._stm, self._ltm]:
                to_remove = [
                    k for k, item in store.items()
                    if (now - item.last_accessed) > max_age_s and item.access_count < 2
                ]
                for key in to_remove:
                    del store[key]
                    removed += 1
        return removed

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "stm_size": len(self._stm),
                "ltm_size": len(self._ltm),
                "total_items": len(self._stm) + len(self._ltm),
                "stm_capacity": self._stm_capacity,
                "ltm_capacity": self._ltm_capacity,
            }


# ============================================================================
# 11. Emergent Behavior Detector
# ============================================================================

@dataclass
class EmergentPattern:
    pattern_type: str  # convergence, oscillation, cascade, clustering
    agents_involved: List[str]
    strength: float
    recommendation: str
    detected_at: float = field(default_factory=time.time)


class EmergentBehaviorDetector:
    """
    Detect and amplify beneficial emergent patterns in swarm activity.

    Monitors agent actions for:
      - Convergence: multiple agents independently reaching similar conclusions
      - Oscillation: agents flip-flopping (negative pattern, dampens)
      - Cascade: one agent's output triggering chain of improvements
      - Clustering: agents naturally specializing into subgroups
    """

    def __init__(self, window_size: int = 50):
        self._events: deque = deque(maxlen=window_size)
        self._patterns: List[EmergentPattern] = []
        self._lock = threading.Lock()

    def record_event(self, event_type: str, agent_id: str, data: Any = None) -> None:
        with self._lock:
            self._events.append({
                "type": event_type, "agent": agent_id,
                "data": data, "time": time.time(),
            })

    def detect_patterns(self) -> List[EmergentPattern]:
        """Scan recent events for emergent patterns."""
        with self._lock:
            events = list(self._events)
        if len(events) < 5:
            return []

        patterns = []

        # Convergence: same event type from multiple agents
        type_agents: Dict[str, Set[str]] = defaultdict(set)
        for e in events[-20:]:
            type_agents[e["type"]].add(e["agent"])
        for event_type, agents in type_agents.items():
            if len(agents) >= 3:
                patterns.append(EmergentPattern(
                    pattern_type="convergence",
                    agents_involved=sorted(agents),
                    strength=len(agents) / 5.0,
                    recommendation="amplify_consensus",
                ))

        # Oscillation: same agent alternating between two states
        agent_sequences: Dict[str, List[str]] = defaultdict(list)
        for e in events[-20:]:
            agent_sequences[e["agent"]].append(e["type"])
        for agent_id, seq in agent_sequences.items():
            if len(seq) >= 4:
                alternating = all(seq[i] != seq[i + 1] for i in range(len(seq) - 1))
                if alternating and len(set(seq)) <= 2:
                    patterns.append(EmergentPattern(
                        pattern_type="oscillation",
                        agents_involved=[agent_id],
                        strength=0.8,
                        recommendation="dampen_oscillation",
                    ))

        # Cascade: sequential events from different agents within short time
        cascades = []
        for i in range(len(events) - 2):
            if (events[i + 1]["time"] - events[i]["time"] < 5.0 and
                events[i]["agent"] != events[i + 1]["agent"]):
                cascades.append(events[i]["agent"])
        if len(cascades) >= 3:
            patterns.append(EmergentPattern(
                pattern_type="cascade",
                agents_involved=list(set(cascades)),
                strength=len(cascades) / 10.0,
                recommendation="encourage_cascade",
            ))

        # Clustering: agents with similar event types grouping
        from collections import Counter
        agent_profiles: Dict[str, Counter] = defaultdict(Counter)
        for e in events:
            agent_profiles[e["agent"]][e["type"]] += 1
        if len(agent_profiles) >= 4:
            profiles_list = list(agent_profiles.items())
            clusters = []
            for i, (a1, p1) in enumerate(profiles_list):
                for a2, p2 in profiles_list[i + 1:]:
                    shared = sum((p1 & p2).values())
                    total = sum((p1 | p2).values())
                    if total > 0 and shared / total > 0.6:
                        clusters.append((a1, a2))
            if clusters:
                all_clustered = set()
                for a, b in clusters:
                    all_clustered.add(a)
                    all_clustered.add(b)
                patterns.append(EmergentPattern(
                    pattern_type="clustering",
                    agents_involved=sorted(all_clustered),
                    strength=len(clusters) / 5.0,
                    recommendation="formalize_specialization",
                ))

        with self._lock:
            self._patterns.extend(patterns)
        return patterns

    def stats(self) -> Dict[str, Any]:
        by_type = defaultdict(int)
        for p in self._patterns:
            by_type[p.pattern_type] += 1
        return {
            "events_recorded": len(self._events),
            "patterns_detected": len(self._patterns),
            "by_type": dict(by_type),
        }


# ============================================================================
# 12. Multi-Agent Debate Arena
# ============================================================================

class DebateFormat(Enum):
    ROUND_ROBIN = "round_robin"
    PANEL = "panel"
    ADVERSARIAL = "adversarial"


@dataclass
class DebateRound:
    round_number: int
    contributions: Dict[str, str]  # agent_id → argument
    moderator_summary: str = ""
    areas_of_agreement: List[str] = field(default_factory=list)
    areas_of_disagreement: List[str] = field(default_factory=list)


@dataclass
class DebateTranscript:
    topic: str
    format: DebateFormat
    participants: List[str]
    moderator_id: str
    rounds: List[DebateRound]
    verdict: str = ""
    quality_scores: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MultiAgentDebateArena:
    """
    N-way structured debate with moderator.

    Unlike the existing DebateArena (1v1 bracket), this supports
    N participants simultaneously with a moderator guiding discussion.

    Formats:
      ROUND_ROBIN: every participant speaks each round
      PANEL: moderator directs questions to specific agents
      ADVERSARIAL: teams assigned pro/con positions
    """

    PARTICIPANT_ROLES = [
        {"name": "innovator", "style": "Focus on novel approaches and creative angles"},
        {"name": "critic", "style": "Challenge assumptions and find weaknesses"},
        {"name": "pragmatist", "style": "Focus on feasibility and practical implementation"},
        {"name": "theorist", "style": "Emphasize theoretical foundations and rigor"},
        {"name": "synthesizer", "style": "Find connections and build on others' ideas"},
    ]

    def __init__(self, max_rounds: int = 3):
        self.max_rounds = max_rounds
        self._transcripts: List[DebateTranscript] = []

    def build_participant_prompt(self, role: Dict, topic: str,
                                  round_num: int, prior_arguments: Dict[str, str]) -> Dict[str, str]:
        """Build prompt for a debate participant."""
        prior = "\n".join(f"  {agent}: {arg[:200]}" for agent, arg in prior_arguments.items())
        return {
            "system": (
                f"You are a {role['name']} in a multi-agent research debate. "
                f"{role['style']}. Be concise but substantive. "
                f"Respond to others' arguments directly.\n\n"
                f"Return JSON: {{\"argument\": \"your position\", "
                f"\"rebuttals\": [\"responses to specific other arguments\"], "
                f"\"key_point\": \"single most important point\"}}"
            ),
            "user": (
                f"DEBATE TOPIC: {topic}\n"
                f"ROUND: {round_num}/{self.max_rounds}\n\n"
                f"PRIOR ARGUMENTS:\n{prior}\n\n"
                f"Present your argument as {role['name']}."
            ),
        }

    def build_moderator_prompt(self, topic: str, round_contributions: Dict[str, str],
                                round_num: int) -> Dict[str, str]:
        """Build prompt for the moderator to summarize a round."""
        contributions = "\n".join(f"  {agent}: {arg[:200]}" for agent, arg in round_contributions.items())
        return {
            "system": (
                "You are an impartial debate moderator. Summarize this round, "
                "identify areas of agreement and disagreement, and pose the "
                "key question for the next round.\n\n"
                "Return JSON: {\"summary\": \"...\", "
                "\"agreements\": [\"...\"], \"disagreements\": [\"...\"], "
                "\"next_question\": \"...\"}"
            ),
            "user": (
                f"TOPIC: {topic}\nROUND {round_num} CONTRIBUTIONS:\n{contributions}"
            ),
        }

    def build_verdict_prompt(self, topic: str, transcript: DebateTranscript) -> Dict[str, str]:
        """Build prompt for final verdict."""
        rounds_text = ""
        for r in transcript.rounds:
            rounds_text += f"\nRound {r.round_number}: {r.moderator_summary[:200]}\n"
        return {
            "system": (
                "You are the chief judge of a research debate. Based on all rounds, "
                "determine which position is strongest and provide your verdict.\n\n"
                "Return JSON: {\"verdict\": \"...\", \"winning_position\": \"...\", "
                "\"quality_scores\": {\"participant_id\": score}, "
                "\"key_insights\": [\"...\"], \"recommendation\": \"...\"}"
            ),
            "user": f"TOPIC: {topic}\nDEBATE TRANSCRIPT:{rounds_text}",
        }

    def record_debate(self, transcript: DebateTranscript) -> None:
        self._transcripts.append(transcript)

    def stats(self) -> Dict[str, Any]:
        return {
            "debates_held": len(self._transcripts),
            "formats_used": [t.format.value for t in self._transcripts[-5:]],
            "avg_rounds": round(
                sum(len(t.rounds) for t in self._transcripts) / max(len(self._transcripts), 1), 1
            ),
            "avg_participants": round(
                sum(len(t.participants) for t in self._transcripts) / max(len(self._transcripts), 1), 1
            ),
        }


# ============================================================================
# Master Swarm Optimizer
# ============================================================================

class SwarmOptimizer:
    """Aggregates all agent swarm optimization techniques."""

    def __init__(self, enable_all: bool = True):
        self.agent_pool = AgentPool() if enable_all else None
        self.message_bus = MessageBus() if enable_all else None
        self.blackboard = SharedBlackboard() if enable_all else None
        self.consensus = SwarmConsensus() if enable_all else None
        self.coordinator = HierarchicalCoordinator() if enable_all else None
        self.router = SpecialistRouter() if enable_all else None
        self.team_former = DynamicTeamFormer(
            pool=self.agent_pool, router=self.router,
        ) if enable_all else None
        self.stigmergy = StigmergyEngine() if enable_all else None
        self.negotiator = AgentNegotiator() if enable_all else None
        self.memory_pool = SwarmMemoryPool() if enable_all else None
        self.emergence = EmergentBehaviorDetector() if enable_all else None
        self.debate = MultiAgentDebateArena() if enable_all else None

    def summary(self) -> Dict[str, Any]:
        result = {}
        if self.agent_pool: result["agent_pool"] = self.agent_pool.stats()
        if self.message_bus: result["message_bus"] = self.message_bus.stats()
        if self.blackboard: result["blackboard"] = self.blackboard.stats()
        if self.consensus: result["consensus"] = self.consensus.stats()
        if self.coordinator: result["coordinator"] = self.coordinator.stats()
        if self.router: result["router"] = self.router.stats()
        if self.team_former: result["team_former"] = self.team_former.stats()
        if self.stigmergy: result["stigmergy"] = self.stigmergy.stats()
        if self.negotiator: result["negotiator"] = self.negotiator.stats()
        if self.memory_pool: result["memory_pool"] = self.memory_pool.stats()
        if self.emergence: result["emergence"] = self.emergence.stats()
        if self.debate: result["debate"] = self.debate.stats()
        return result

    def shutdown(self) -> None:
        if self.agent_pool:
            self.agent_pool.shutdown()
