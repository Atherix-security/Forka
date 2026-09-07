from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from numbers import Real
from typing import Any, Callable, Iterable, Protocol

from .identity import state_fingerprint
from .metadata import ReproducibilityRecord, capture_record
from .pruning import MinProbability, PruningStrategy, TopK, select_branches


@dataclass(frozen=True)
class State:
    data: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    terminal: bool = False

    def with_data(self, **updates: Any) -> "State":
        return replace(self, data={**self.data, **updates})

    def with_score(self, score: float) -> "State":
        return replace(self, score=score)

    def stop(self) -> "State":
        return replace(self, terminal=True)

    def fingerprint(self) -> str:
        """Deterministic identity of supported state data, score and terminal flag."""
        return state_fingerprint(self)


class Evaluator(Protocol):
    """Domain score applied before pruning; higher scores win probability ties."""

    def __call__(self, state: State) -> float: ...


class TerminationReason(str, Enum):
    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    MAX_RUNTIME = "max_runtime"
    MAX_BRANCHES = "max_branches"
    MIN_PROBABILITY = "min_probability"
    NO_BRANCHES = "no_branches"
    CUSTOM_PRUNING = "custom_pruning"


@dataclass(frozen=True)
class SimulationStatistics:
    """Counts of generated candidates and retained frontiers; see architecture doc."""

    branches_explored: int = 0
    branches_pruned: int = 0
    peak_active_branches: int = 0
    retained_probability_mass: float = 0.0
    runtime_seconds: float = 0.0
    # New observational metrics preserve the v0.2 five-field equality contract.
    states_visited: int = field(default=0, compare=False)
    transitions_generated: int = field(default=0, compare=False)
    branches_retained: int = field(default=0, compare=False)
    branches_merged: int = field(default=0, compare=False)
    states_deduplicated: int = field(default=0, compare=False)
    absorbed_probability_mass: float = field(default=0.0, compare=False)
    removed_probability_mass: float = field(default=0.0, compare=False)

    @property
    def branches_generated(self) -> int:
        return self.branches_explored

    @property
    def transitions_per_second(self) -> float:
        return (
            self.transitions_generated / self.runtime_seconds
            if self.runtime_seconds > 0
            else 0.0
        )


@dataclass(frozen=True)
class Transition:
    probability: float
    state: State
    label: str = ""


@dataclass
class Branch:
    state: State
    probability: float = 1.0
    history: tuple[str, ...] = ()
    steps: int = 0
    path_count: int = 1


@dataclass(frozen=True)
class SimulationConfig:
    max_steps: int = 20
    max_branches: int = 128
    max_runtime_seconds: float | None = None
    min_probability: float = 1e-9
    seed: int | None = None
    deduplication: str = "none"

    def __post_init__(self):
        if self.deduplication not in ("none", "state_markov"):
            raise ValueError(
                "deduplication must be 'none' or the explicit 'state_markov' contract"
            )
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer or None")
        if not isinstance(self.max_steps, int) or self.max_steps < 0:
            raise ValueError("max_steps must be a nonnegative integer")
        if not isinstance(self.max_branches, int) or self.max_branches < 1:
            raise ValueError("max_branches must be a positive integer")
        if (
            not math.isfinite(self.min_probability)
            or not 0 <= self.min_probability <= 1
        ):
            raise ValueError("min_probability must be finite and between 0 and 1")
        if self.max_runtime_seconds is not None and (
            not math.isfinite(self.max_runtime_seconds) or self.max_runtime_seconds < 0
        ):
            raise ValueError("max_runtime_seconds must be finite and nonnegative")


@dataclass
class SimulationResult:
    branches: list[Branch]
    elapsed_seconds: float
    steps_executed: int
    truncated: bool = False
    termination_reasons: tuple[TerminationReason, ...] = ()
    statistics: SimulationStatistics = field(default_factory=SimulationStatistics)
    reproducibility: ReproducibilityRecord | None = field(default=None, compare=False)

    def best(self, key: Callable[[Branch], float] | None = None) -> Branch | None:
        if not self.branches:
            return None
        key = key or (lambda b: b.state.score)
        return max(self.branches, key=key)


StepFn = Callable[
    [State, random.Random], Iterable[Transition | tuple[float, State, str]]
]


class Simulation:
    def __init__(
        self,
        step: StepFn,
        config: SimulationConfig | None = None,
        *,
        evaluator: Evaluator | None = None,
        pruning: PruningStrategy | None = None,
    ):
        self.step = step
        self.config = config or SimulationConfig()
        self.evaluator = evaluator
        self.pruning = pruning

    def _evaluate(self, state: State) -> State:
        if self.evaluator is None:
            return state
        score = self.evaluator(state)
        if (
            isinstance(score, bool)
            or not isinstance(score, Real)
            or not math.isfinite(score)
        ):
            raise ValueError("evaluator must return a finite real score")
        return state.with_score(float(score))

    def run(self, initial: State) -> SimulationResult:
        rng = random.Random(self.config.seed)
        started = time.monotonic()
        active = [Branch(self._evaluate(initial))]
        truncated = False
        executed = 0
        explored = 0
        pruned = 0
        peak = 1
        runtime_exhausted = False
        reasons: list[TerminationReason] = []
        visited = 0
        generated = 0
        merged = 0
        deduplicated = 0
        absorbed_mass = 0.0
        removed_mass = 0.0
        merging = self.config.deduplication == "state_markov"
        for _ in range(self.config.max_steps):
            if all(b.state.terminal for b in active):
                break
            if (
                self.config.max_runtime_seconds is not None
                and time.monotonic() - started >= self.config.max_runtime_seconds
            ):
                truncated = True
                runtime_exhausted = True
                break
            next_branches: list[Branch] = []
            for index, branch in enumerate(active):
                if branch.state.terminal:
                    next_branches.append(branch)
                    continue
                if (
                    self.config.max_runtime_seconds is not None
                    and time.monotonic() - started >= self.config.max_runtime_seconds
                ):
                    next_branches.extend(active[index:])
                    truncated = True
                    runtime_exhausted = True
                    break
                visited += 1
                transitions = list(self.step(branch.state, rng))
                generated += len(transitions)
                if not transitions:
                    next_branches.append(
                        Branch(
                            self._evaluate(branch.state.stop()),
                            branch.probability,
                            branch.history,
                            branch.steps,
                            branch.path_count,
                        )
                    )
                    continue
                transitions = [
                    t if isinstance(t, Transition) else Transition(*t)
                    for t in transitions
                ]
                if any(
                    not math.isfinite(t.probability) or t.probability < 0
                    for t in transitions
                ):
                    raise ValueError(
                        "transition probabilities must be finite and nonnegative"
                    )
                scale = max(t.probability for t in transitions)
                if scale <= 0:
                    raise ValueError(
                        "transition probabilities must sum to a positive value"
                    )
                total = sum(t.probability / scale for t in transitions)
                for t in transitions:
                    if t.probability > 0:
                        explored += 1
                    p = branch.probability * ((t.probability / scale) / total)
                    if p > 0 and (
                        merging
                        or self.pruning is not None
                        or p >= self.config.min_probability
                    ):
                        next_branches.append(
                            Branch(
                                self._evaluate(t.state),
                                p,
                                branch.history + (t.label,),
                                branch.steps + 1,
                                branch.path_count,
                            )
                        )
                    elif t.probability > 0:
                        truncated = True
                        pruned += 1
                        removed_mass += p
                        if TerminationReason.MIN_PROBABILITY not in reasons:
                            reasons.append(TerminationReason.MIN_PROBABILITY)
            if merging:
                next_branches, count, groups, mass = _merge_frontier(next_branches)
                merged += count
                deduplicated += groups
                absorbed_mass += mass
            if self.pruning is not None:
                strategies = [self.pruning]
            else:
                strategies = (
                    [MinProbability(self.config.min_probability)] if merging else []
                )
                strategies.append(TopK(self.config.max_branches))
            for strategy in strategies:
                next_branches, count, mass = select_branches(next_branches, strategy)
                if count:
                    truncated = True
                    pruned += count
                    removed_mass += mass
                    reason = (
                        TerminationReason.MIN_PROBABILITY
                        if isinstance(strategy, MinProbability)
                        else TerminationReason.MAX_BRANCHES
                        if isinstance(strategy, TopK)
                        else TerminationReason.CUSTOM_PRUNING
                    )
                    if reason not in reasons:
                        reasons.append(reason)
            next_branches.sort(
                key=lambda b: (b.probability, b.state.score), reverse=True
            )
            active = next_branches
            peak = max(peak, len(active))
            executed += 1
            if not active:
                break
        truncated = truncated or any(not b.state.terminal for b in active)
        if runtime_exhausted:
            reasons.append(TerminationReason.MAX_RUNTIME)
        elif not active:
            reasons.append(TerminationReason.NO_BRANCHES)
        elif any(not b.state.terminal for b in active):
            reasons.append(TerminationReason.MAX_STEPS)
        else:
            reasons.append(TerminationReason.COMPLETED)
        elapsed = time.monotonic() - started
        statistics = SimulationStatistics(
            branches_explored=explored,
            branches_pruned=pruned,
            peak_active_branches=peak,
            retained_probability_mass=sum(b.probability for b in active),
            runtime_seconds=elapsed,
            states_visited=visited,
            transitions_generated=generated,
            branches_retained=len(active),
            branches_merged=merged,
            states_deduplicated=deduplicated,
            absorbed_probability_mass=absorbed_mass,
            removed_probability_mass=removed_mass,
        )
        return SimulationResult(
            active,
            elapsed,
            executed,
            truncated,
            tuple(reasons),
            statistics,
            capture_record(
                self.config, self.step, pruning=self.pruning, evaluator=self.evaluator
            ),
        )


def _merge_frontier(branches: list[Branch]):
    """Merge only exact state/depth matches under the explicit caller contract."""
    groups: dict[tuple[str, int], list[Branch]] = {}
    for branch in branches:
        groups.setdefault((branch.state.fingerprint(), branch.steps), []).append(branch)
    retained = []
    merged = 0
    duplicate_groups = 0
    absorbed = []
    for group in groups.values():
        representative = group[0]
        if len(group) > 1:
            merged += len(group) - 1
            duplicate_groups += 1
            absorbed.extend(b.probability for b in group[1:])
            representative = replace(
                representative,
                probability=math.fsum(b.probability for b in group),
                path_count=sum(b.path_count for b in group),
            )
        retained.append(representative)
    return retained, merged, duplicate_groups, math.fsum(absorbed)
