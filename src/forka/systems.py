"""Explicit adapters around trusted Python systems; no Agent requirement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Protocol, TypeVar

from .core import State, Transition
from .runtime import Model

Input = TypeVar("Input")
Output = TypeVar("Output")
InputContra = TypeVar("InputContra", contravariant=True)
OutputCo = TypeVar("OutputCo", covariant=True)


class SystemUnderTest(Protocol[InputContra, OutputCo]):
    def run(self, value: InputContra) -> OutputCo: ...


@dataclass(frozen=True)
class CallableSystem(Generic[Input, Output]):
    function: Callable[[Input], Output]

    def run(self, value: Input) -> Output:
        return self.function(value)


@dataclass(frozen=True)
class SystemModel(Model, Generic[Input, Output]):
    system: SystemUnderTest[Input, Output] | Callable[[Input], Output]
    input_adapter: Callable[[State], Input]
    output_adapter: Callable[[State, Output], State]

    def step(self, state, rng):
        value = self.input_adapter(state)
        if callable(self.system):
            output = self.system(value)
        else:
            output = self.system.run(value)
        next_state = self.output_adapter(state, output)
        if not isinstance(next_state, State):
            raise TypeError("output_adapter must return a State")
        return [Transition(1.0, next_state, "system")]
