import random
import runpy
from pathlib import Path

import pytest

from forka import (
    Action,
    Agent,
    Model,
    Observation,
    Scenario,
    Simulation,
    SimulationConfig,
    State,
    StateAdapter,
    TerminationReason,
    Transition,
)


class Growth:
    """Structural Model conformance, without inheriting from Forka classes."""

    def step(self, state, rng):
        return [(1, state.with_data(x=state.data["x"] + rng.random()).stop(), "grow")]


def test_generic_scenario_matches_legacy_kernel_and_repeats():
    model: Model = Growth()
    initial = State({"x": 2})
    config = SimulationConfig(seed=17)
    scenario = Scenario(model, initial, config)
    first = scenario.run()
    assert first.branches == Simulation(model.step, config).run(initial).branches
    assert first.branches == scenario.run().branches
    assert first.statistics.branches_explored == 1
    assert first.termination_reasons == (TerminationReason.COMPLETED,)
    assert initial == State({"x": 2})


class CounterEnvironment:
    def __init__(self):
        self.applied = []  # Test instrumentation only, not evolving world state.

    def observe(self, state, actor):
        return Observation({"visible": state.data["x"], "actor": actor})

    def apply(self, state, actor, action, rng):
        self.applied.append((actor, action.name))
        return state.with_data(x=state.data["x"] + action.data["delta"]).stop()


def test_agent_is_model_with_weighted_actions_and_isolated_siblings():
    env = CounterEnvironment()

    def policy(observation, rng):
        assert observation == Observation({"visible": 10, "actor": "counter"})
        yield Action("up", {"delta": 2}, 3)
        yield Action("down", {"delta": -1}, 1)
        yield Action("impossible", {"delta": 99}, 0)

    agent = Agent("counter", env, policy)
    assert Model in Agent.__mro__
    initial = State({"x": 10})
    result = Scenario(agent, initial).run()
    assert [b.state.data["x"] for b in result.branches] == [12, 9]
    assert [b.probability for b in result.branches] == [0.75, 0.25]
    assert [b.history for b in result.branches] == [("counter:up",), ("counter:down",)]
    assert env.applied == [("counter", "up"), ("counter", "down")]
    assert initial.data == {"x": 10}


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf"), 0])
def test_invalid_action_distribution_does_not_apply_world_changes(weight):
    env = CounterEnvironment()
    agent = Agent("a", env, lambda obs, rng: [Action("bad", probability=weight)])
    with pytest.raises(ValueError, match="action probabilities"):
        Scenario(agent, State({"x": 0})).run()
    assert env.applied == []


def test_empty_policy_and_seeded_policy():
    env = CounterEnvironment()
    empty = Scenario(Agent("a", env, lambda o, r: []), State({"x": 0})).run()
    assert empty.branches[0].state.terminal
    assert empty.statistics.branches_explored == 0
    assert empty.termination_reasons == (TerminationReason.COMPLETED,)
    agent = Agent("a", env, lambda o, r: [Action("random", {"delta": r.random()})])
    scenario = Scenario(agent, State({"x": 0}), SimulationConfig(seed=5))
    assert scenario.run().branches == scenario.run().branches
    assert (
        scenario.run().branches
        != Scenario(agent, State({"x": 0}), SimulationConfig(seed=6)).run().branches
    )


def test_environment_receives_seeded_rng():
    class RandomEnvironment(CounterEnvironment):
        def apply(self, state, actor, action, rng):
            return state.with_data(x=rng.random()).stop()

    agent = Agent("a", RandomEnvironment(), lambda o, r: [Action("sample")])
    result = Scenario(agent, State({"x": 0}), SimulationConfig(seed=8)).run()
    assert result.best().state.data["x"] == random.Random(8).random()


def test_evaluator_changes_score_tie_break_and_best_without_mutation():
    states = [State({"x": i}, score=99, terminal=True) for i in range(3)]

    class Choices:
        def step(self, state, rng):
            return [Transition(1, child) for child in states]

    result = Scenario(
        Choices(),
        config=SimulationConfig(max_branches=1),
        evaluator=lambda s: s.data.get("x", 0),
    ).run()
    assert result.best().state.data["x"] == 2
    assert result.best().state.score == 2
    assert all(s.score == 99 for s in states)
    assert result.statistics.branches_pruned == 2


def test_evaluator_preserves_probability_first_ranking():
    sim = Simulation(
        lambda s, r: [
            (3, State(score=1, terminal=True), "likely"),
            (1, State(score=100, terminal=True), "unlikely"),
        ],
        SimulationConfig(max_branches=1),
        evaluator=lambda s: s.score,
    )
    assert sim.run(State()).best().history == ("likely",)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), "1", True, None])
def test_invalid_evaluator_scores(score):
    with pytest.raises(ValueError, match="finite real score"):
        Scenario(Growth(), State({"x": 0}), evaluator=lambda s: score).run()


def test_evaluator_initial_terminal_and_empty_successors():
    result = Scenario(
        Growth(), State({"x": 2}, terminal=True), evaluator=lambda s: 4
    ).run()
    assert result.best().state.score == 4
    result = Simulation(
        lambda s, r: [], evaluator=lambda s: 1 if s.terminal else 0
    ).run(State())
    assert result.best().state.score == 1


def test_domain_errors_propagate():
    def fail(state):
        raise LookupError("domain error")

    with pytest.raises(LookupError, match="domain error"):
        Scenario(Growth(), evaluator=fail).run()


def test_model_and_environment_errors_propagate():
    class BrokenModel:
        def step(self, state, rng):
            raise RuntimeError("model failure")

    with pytest.raises(RuntimeError, match="model failure"):
        Scenario(BrokenModel()).run()

    class BrokenWorld(CounterEnvironment):
        def apply(self, state, actor, action, rng):
            raise RuntimeError("environment failure")

    agent = Agent("broken", BrokenWorld(), lambda o, r: [Action("go")])
    with pytest.raises(RuntimeError, match="environment failure"):
        Scenario(agent, State({"x": 0})).run()


def test_evaluator_validates_generated_state_and_skips_probability_rejections():
    def score(state):
        return state.data.get("score", 0)

    with pytest.raises(ValueError, match="finite real score"):
        Simulation(
            lambda s, r: [(1, State({"score": float("nan")}), "bad")], evaluator=score
        ).run(State())
    result = Simulation(
        lambda s, r: [
            (0.01, State({"score": float("nan")}), "pruned"),
            (0.99, State({"score": 2}, terminal=True), "kept"),
        ],
        SimulationConfig(min_probability=0.1),
        evaluator=score,
    ).run(State())
    assert result.best().state.score == 2


def test_external_adapter_roundtrip_without_library_dependency():
    class VectorAdapter:
        def to_state(self, value):
            return State({"coordinates": list(value)})

        def from_state(self, state):
            return tuple(state.data["coordinates"])

    adapter: StateAdapter[tuple[float, ...]] = VectorAdapter()
    external = (1.0, 2.5)
    state = adapter.to_state(external)
    assert adapter.from_state(state) == external
    assert len(state.fingerprint()) == 64


@pytest.mark.parametrize("example", ["attacker_defender", "population_growth"])
def test_examples_are_complete_reproducible_and_agent_optional(example):
    namespace = runpy.run_path(
        str(Path(__file__).parents[1] / "examples" / f"{example}.py")
    )
    scenario = namespace["build_scenario"]()
    first = scenario.run()
    assert first.branches == scenario.run().branches
    assert first.termination_reasons == (TerminationReason.COMPLETED,)
    assert not first.truncated
    assert first.statistics.retained_probability_mass == pytest.approx(1)
    assert all(b.state.terminal for b in first.branches)
    if example == "attacker_defender":
        assert len(first.branches) == 16
        assert (
            scenario.model.attacker.environment is scenario.model.defender.environment
        )
        assert all(
            tuple(label.split(":")[0] for label in b.history)
            == ("attacker", "defender", "attacker", "defender")
            for b in first.branches
        )
    else:
        assert not isinstance(scenario.model, Agent)
        assert len(first.branches) == 8
        assert all(b.state.data["population"] > 100 for b in first.branches)
        assert first.branches != namespace["build_scenario"](seed=9).run().branches
