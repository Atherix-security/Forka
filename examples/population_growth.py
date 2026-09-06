"""Illustrative population model with weather branches; not a calibrated forecast."""

from forka import Model, Scenario, SimulationConfig, State, Transition


class PopulationModel(Model):
    """A scientific-style model with no Agent, Action or Environment requirement."""

    def step(self, state, rng):
        year = state.data["year"] + 1
        # Small seeded variation plus explicitly enumerated weather uncertainty.
        growth = rng.uniform(0.08, 0.12)
        for label, weight, effect in [("dry", 0.4, -0.04), ("wet", 0.6, 0.03)]:
            population = state.data["population"] * (1 + growth + effect)
            yield Transition(
                weight,
                State({"population": population, "year": year}, terminal=year >= 3),
                label,
            )


def build_scenario(seed=7):
    return Scenario(
        model=PopulationModel(),
        initial_state=State({"population": 100.0, "year": 0}),
        config=SimulationConfig(max_steps=3, max_branches=16, seed=seed),
        evaluator=lambda state: state.data["population"],
    )


if __name__ == "__main__":
    result = build_scenario().run()
    expected_population = sum(
        branch.probability * branch.state.data["population"]
        for branch in result.branches
    )
    print(f"Expected population after three years: {expected_population:.2f}")
    print("Best state fingerprint:", result.best().state.fingerprint())
    print("Statistics:", result.statistics)
