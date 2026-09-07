"""Small interfaces for composing domain models and optional agents."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Protocol, TypeVar

from .core import (
    Evaluator,
    Simulation,
    SimulationConfig,
    SimulationResult,
    State,
    Transition,
)
from .metadata import qualified_name
from .pruning import PruningStrategy


class Model(Protocol):
    """A domain evolution rule. Implementing ``step`` is sufficient."""

    def step(
        self, state: State, rng: random.Random
    ) -> Iterable[Transition | tuple[float, State, str]]: ...


@dataclass(frozen=True)
class Observation:
    """Environment-defined policy input; treat nested data as immutable."""

    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    """Named intent and relative probability weight, normalized by the kernel."""

    name: str
    data: dict[str, Any] = field(default_factory=dict)
    probability: float = 1.0


class Environment(Protocol):
    """World rules and views; evolving world data belongs in each branch's State."""

    def observe(self, state: State, actor: str) -> Observation: ...

    def apply(
        self, state: State, actor: str, action: Action, rng: random.Random
    ) -> State: ...


@dataclass(frozen=True)
class Agent(Model):
    """A Model composed from a policy and environment; no inference required.

    Policies enumerate alternative actions, each applied to the same input state.
    Implementations must not mutate that state or keep evolving state on services.
    """

    name: str
    environment: Environment
    policy: Callable[[Observation, random.Random], Iterable[Action]]

    def step(self, state: State, rng: random.Random) -> Iterable[Transition]:
        observation = self.environment.observe(state, self.name)
        actions = list(self.policy(observation, rng))
        if any(not math.isfinite(a.probability) or a.probability < 0 for a in actions):
            raise ValueError("action probabilities must be finite and nonnegative")
        if actions and not any(a.probability > 0 for a in actions):
            raise ValueError("action probabilities must sum to a positive value")
        for action in actions:
            if action.probability > 0:
                yield Transition(
                    action.probability,
                    self.environment.apply(state, self.name, action, rng),
                    f"{self.name}:{action.name}",
                )


@dataclass(frozen=True)
class Scenario:
    """A reusable local run specification independent of any transport or UI."""

    model: Model
    initial_state: State = field(default_factory=State)
    config: SimulationConfig = field(default_factory=SimulationConfig)
    evaluator: Evaluator | None = None
    pruning: PruningStrategy | None = None
    identifier: str | None = None

    def run(self) -> SimulationResult:
        result = Simulation(
            self.model.step, self.config, evaluator=self.evaluator, pruning=self.pruning
        ).run(self.initial_state)
        result.reproducibility = replace(
            result.reproducibility,
            model_identifier=qualified_name(self.model),
            scenario_identifier=self.identifier or qualified_name(self.model),
        )
        return result


ExternalState = TypeVar("ExternalState")


class StateAdapter(Protocol[ExternalState]):
    """Boundary for external data; library-specific imports belong in adapters."""

    def to_state(self, value: ExternalState) -> State: ...

    def from_state(self, state: State) -> ExternalState: ...
