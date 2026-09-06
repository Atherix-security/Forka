from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from numbers import Real
from typing import Any, Callable, Iterable, Protocol

from .identity import state_fingerprint


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


@dataclass(frozen=True)
class SimulationStatistics:
    """Counts of generated candidates and retained frontiers; see architecture doc."""

    branches_explored: int = 0
    branches_pruned: int = 0
    peak_active_branches: int = 0
    retained_probability_mass: float = 0.0
    runtime_seconds: float = 0.0


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


@dataclass(frozen=True)
class SimulationConfig:
    max_steps: int = 20
    max_branches: int = 128
    max_runtime_seconds: float | None = None
    min_probability: float = 1e-9
    seed: int | None = None

    def __post_init__(self):
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
    ):
        self.step = step
        self.config = config or SimulationConfig()
        self.evaluator = evaluator

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
                transitions = list(self.step(branch.state, rng))
                if not transitions:
                    next_branches.append(
                        Branch(
                            self._evaluate(branch.state.stop()),
                            branch.probability,
                            branch.history,
                            branch.steps,
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
                    if p > 0 and p >= self.config.min_probability:
                        next_branches.append(
                            Branch(
                                self._evaluate(t.state),
                                p,
                                branch.history + (t.label,),
                                branch.steps + 1,
                            )
                        )
                    elif t.probability > 0:
                        truncated = True
                        pruned += 1
                        if TerminationReason.MIN_PROBABILITY not in reasons:
                            reasons.append(TerminationReason.MIN_PROBABILITY)
            next_branches.sort(
                key=lambda b: (b.probability, b.state.score), reverse=True
            )
            if len(next_branches) > self.config.max_branches:
                truncated = True
                pruned += len(next_branches) - self.config.max_branches
                if TerminationReason.MAX_BRANCHES not in reasons:
                    reasons.append(TerminationReason.MAX_BRANCHES)
                next_branches = next_branches[: self.config.max_branches]
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
        )
        return SimulationResult(
            active, elapsed, executed, truncated, tuple(reasons), statistics
        )
