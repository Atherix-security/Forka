"""Exact Markov-state merging preserves mass without retaining every path."""

from forka import Model, NoPruning, Scenario, SimulationConfig, State, Transition


class RandomWalk(Model):
    def step(self, state, rng):
        tick = state.data["tick"] + 1
        for direction in (-1, 1):
            yield Transition(
                0.5,
                State(
                    {"tick": tick, "x": state.data["x"] + direction}, terminal=tick == 6
                ),
                str(direction),
            )


def build_scenario(deduplication="state_markov"):
    # No RNG draws or hidden state: future evolution is a function of this state.
    return Scenario(
        RandomWalk(),
        State({"tick": 0, "x": 0}),
        SimulationConfig(seed=7, deduplication=deduplication),
        pruning=NoPruning(),
    )


if __name__ == "__main__":
    for mode in ("none", "state_markov"):
        result = build_scenario(mode).run()
        print(
            mode,
            "retained:",
            len(result.branches),
            "merged:",
            result.statistics.branches_merged,
            "mass:",
            round(result.statistics.retained_probability_mass, 6),
            "represented paths:",
            sum(b.path_count for b in result.branches),
        )
