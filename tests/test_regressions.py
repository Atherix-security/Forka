from unittest.mock import patch

import pytest

from forka import (
    ComputeBudget,
    IntelligenceRouter,
    MemoryCache,
    RouteRequest,
    Simulation,
    SimulationConfig,
    State,
)


def test_seed_repeats_on_same_simulation():
    def step(state, rng):
        return [(1, state.with_data(value=rng.random()), "draw")]

    sim = Simulation(step, SimulationConfig(max_steps=4, seed=42))
    first = sim.run(State())
    assert first.branches == sim.run(State()).branches
    assert (
        first.branches
        != Simulation(step, SimulationConfig(max_steps=4, seed=43))
        .run(State())
        .branches
    )
    assert first.truncated


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf"), -float("inf")])
def test_invalid_probability(weight):
    with pytest.raises(ValueError):
        Simulation(lambda s, r: [(weight, s, "bad"), (2, s, "ok")]).run(State())


def test_large_weights_normalize_without_overflow():
    result = Simulation(
        lambda s, r: [(1e308, s.stop(), "a"), (1e308, s.stop(), "b")]
    ).run(State())
    assert [b.probability for b in result.branches] == [0.5, 0.5]
    assert not result.truncated


def test_zero_weights_and_threshold_pruning():
    with pytest.raises(ValueError):
        Simulation(lambda s, r: [(0, s, "zero")]).run(State())
    result = Simulation(
        lambda s, r: [(0, s, "zero"), (1, s.stop(), "one")],
        SimulationConfig(min_probability=0),
    ).run(State())
    assert len(result.branches) == 1
    assert not result.truncated
    result = Simulation(
        lambda s, r: [(0.1, s.stop(), "a"), (0.9, s.stop(), "b")],
        SimulationConfig(min_probability=0.2),
    ).run(State())
    assert result.truncated
    assert result.branches[0].probability == pytest.approx(0.9)


@pytest.mark.parametrize(
    "config",
    [
        {"max_steps": -1},
        {"max_steps": 1.5},
        {"max_branches": 0},
        {"max_branches": 2.5},
        {"min_probability": -1},
        {"min_probability": float("nan")},
        {"min_probability": 2},
        {"max_runtime_seconds": -1},
        {"max_runtime_seconds": float("inf")},
    ],
)
def test_invalid_config(config):
    with pytest.raises(ValueError):
        SimulationConfig(**config)


def test_terminal_empty_and_zero_step_runs():
    def forbidden(s, r):
        raise AssertionError("must not step")

    assert not Simulation(forbidden).run(State(terminal=True)).truncated
    result = Simulation(lambda s, r: []).run(State())
    assert result.branches[0].state.terminal and not result.truncated
    result = Simulation(forbidden, SimulationConfig(max_steps=0)).run(State())
    assert result.truncated and result.steps_executed == 0
    result = Simulation(forbidden, SimulationConfig(max_runtime_seconds=0)).run(State())
    assert result.truncated and result.steps_executed == 0


def test_runtime_checked_between_branches():
    clock = [0.0]
    calls = []

    def step(state, rng):
        calls.append(state.data)
        if not state.data:
            return [(1, State({"id": i}), str(i)) for i in range(3)]
        clock[0] += 2
        return [(1, state.stop(), "done")]

    with patch("forka.core.time.monotonic", side_effect=lambda: clock[0]):
        result = Simulation(step, SimulationConfig(max_runtime_seconds=1)).run(State())
    assert len(calls) == 2
    assert len(result.branches) == 3
    assert sum(b.probability for b in result.branches) == pytest.approx(1)
    assert result.truncated


def test_state_helpers_and_best():
    state = State({"x": 1})
    changed = state.with_data(x=2).with_score(3).stop()
    assert state.data == {"x": 1} and not state.terminal
    assert changed.score == 3 and changed.terminal
    result = Simulation(
        lambda s, r: [
            (1, State(score=1, terminal=True), "a"),
            (1, State(score=2, terminal=True), "b"),
        ]
    ).run(state)
    assert result.best().state.score == 2
    assert result.best(key=lambda b: -b.state.score).state.score == 1


def test_budget_exhaustion_is_atomic():
    budget = ComputeBudget(max_model_calls=2, max_cost=0.5)
    budget.spend(0.25)
    with pytest.raises(RuntimeError):
        budget.spend(0.5)
    assert (budget.model_calls, budget.cost) == (1, 0.25)
    budget.spend(0.25)
    assert not budget.can_spend()
    assert not ComputeBudget(max_model_calls=0).can_spend()


@pytest.mark.parametrize("cost", [-1, float("nan"), float("inf")])
def test_invalid_budget_cost(cost):
    with pytest.raises(ValueError):
        ComputeBudget(max_cost=cost)
    budget = ComputeBudget()
    with pytest.raises(ValueError):
        budget.spend(cost)
    assert (budget.model_calls, budget.cost) == (0, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_model_calls": -1},
        {"model_calls": 0.5},
        {"max_model_calls": 1, "model_calls": 2},
        {"max_cost": 1, "cost": 2},
    ],
)
def test_invalid_budget_counters(kwargs):
    with pytest.raises(ValueError):
        ComputeBudget(**kwargs)


def test_cache_keys_expiry_and_replacement():
    cache = MemoryCache()
    assert cache.key({"a": 1, "b": 2}) == cache.key({"b": 2, "a": 1})
    assert cache.get("missing") is None
    with patch("forka.cache.time.monotonic", return_value=10):
        cache.set("key", "first")
    with patch("forka.cache.time.monotonic", return_value=11):
        assert cache.get("key", ttl=2) == "first"
        assert cache.get("key", ttl=1) is None
        cache.set("key", "second")
        assert cache.get("key") == "second"
        assert cache.get("key", ttl=0) is None
    with pytest.raises(ValueError):
        cache.get("key", ttl=-1)


def test_router_callable_provider_and_fallback():
    router = IntelligenceRouter()
    with pytest.raises(RuntimeError):
        router.route(RouteRequest("hello"))
    router.register("rules", lambda prompt: prompt.upper())

    class Local:
        def complete(self, prompt, **kwargs):
            return "local: " + prompt

    router.register("local", Local())
    assert router.route(RouteRequest("hello")) == "HELLO"
    assert router.route(RouteRequest("hello"), preferred="local") == "local: hello"
    assert router.route(RouteRequest("hello"), preferred="missing") == "HELLO"


def test_terminal_state_needs_no_runtime():
    def forbidden(s, r):
        raise AssertionError("must not step")

    result = Simulation(forbidden, SimulationConfig(max_runtime_seconds=0)).run(
        State(terminal=True)
    )
    assert not result.truncated and result.steps_executed == 0
