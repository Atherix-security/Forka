from forka import Simulation, SimulationConfig, State, Transition


def test_branching_and_normalization():
    def step(state, rng):
        return [
            Transition(2, State({"x": 1}, score=1, terminal=True), "a"),
            Transition(1, State({"x": -1}, terminal=True), "b"),
        ]

    result = Simulation(step).run(State())
    assert len(result.branches) == 2
    assert abs(sum(b.probability for b in result.branches) - 1.0) < 1e-9
    assert result.best().state.data["x"] == 1


def test_branch_budget_prunes():
    def step(state, rng):
        return [(1, State({"x": i}), str(i)) for i in range(10)]

    result = Simulation(step, SimulationConfig(max_steps=1, max_branches=3)).run(
        State()
    )
    assert len(result.branches) == 3 and result.truncated
