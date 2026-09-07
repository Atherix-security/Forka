"""Small, public, CPU-only workloads; no external data or service calls."""

import math
from dataclasses import dataclass

from forka import State, Transition


@dataclass(frozen=True)
class LatticeModel:
    depth: int = 7
    divergent: bool = False

    def step(self, state, rng):
        tick = state.data["tick"] + 1
        for label, delta in enumerate((-1, 0, 0, 1)):
            data = {"tick": tick, "x": state.data["x"] + delta}
            if self.divergent:
                data["path"] = state.data.get("path", []) + [label]
            yield Transition(0.25, State(data, terminal=tick >= self.depth), str(label))


@dataclass(frozen=True)
class CpuModel:
    work: int = 10000

    def step(self, state, rng):
        # Fixed deterministic CPU work, then one seeded observation per run.
        value = sum(math.sin(i) * math.cos(i) for i in range(self.work)) + rng.random()
        return [Transition(1, State({"value": value}, terminal=True), "compute")]
