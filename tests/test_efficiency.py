import math
from dataclasses import asdict

import pytest

from forka import (
    MinProbability,
    NoPruning,
    Scenario,
    Simulation,
    SimulationConfig,
    State,
    TopK,
)
from forka import TerminationReason as Reason


def choices(state, rng):
    return [
        (0.2, State({"x": 1}, terminal=True), "a"),
        (0.15, State({"x": 1}, terminal=True), "b"),
        (0.65, State({"x": 2}, terminal=True), "c"),
    ]


def test_explicit_equivalence_combines_mass_and_provenance():
    result = Simulation(choices, SimulationConfig(deduplication="state_markov")).run(
        State()
    )
    assert [b.probability for b in result.branches] == pytest.approx([0.65, 0.35])
    merged = result.branches[1]
    assert merged.history == ("a",) and merged.path_count == 2
    stats = result.statistics
    assert (
        stats.branches_generated,
        stats.branches_merged,
        stats.states_deduplicated,
    ) == (3, 1, 1)
    assert stats.absorbed_probability_mass == pytest.approx(0.15)
    assert stats.retained_probability_mass == pytest.approx(1)
    assert stats.removed_probability_mass == 0
    assert not result.truncated and result.termination_reasons == (Reason.COMPLETED,)


def test_merging_disabled_by_default():
    result = Simulation(choices).run(State())
    assert len(result.branches) == 3
    assert result.statistics.branches_merged == 0
    assert all(b.path_count == 1 for b in result.branches)


def test_merge_precedes_threshold_and_cap_when_enabled():
    config = SimulationConfig(min_probability=0.3, deduplication="state_markov")
    result = Simulation(choices, config).run(State())
    assert len(result.branches) == 2 and result.statistics.branches_pruned == 0
    plain = Simulation(choices, SimulationConfig(min_probability=0.3)).run(State())
    assert plain.statistics.removed_probability_mass == pytest.approx(0.35)
    capped = Simulation(choices, config, pruning=TopK(1)).run(State())
    assert capped.statistics.removed_probability_mass == pytest.approx(0.35)


def test_repeated_merge_path_counts_and_event_mass():
    def step(state, rng):
        tick = state.data.get("tick", 0) + 1
        return [
            (1, State({"tick": tick}, terminal=tick == 3), label)
            for label in ("a", "b")
        ]

    result = Simulation(step, SimulationConfig(deduplication="state_markov")).run(
        State()
    )
    assert result.best().path_count == 8
    assert result.best().history == ("a", "a", "a")
    stats = result.statistics
    assert (
        stats.states_visited,
        stats.transitions_generated,
        stats.branches_merged,
    ) == (3, 6, 3)
    assert stats.states_deduplicated == 3 and stats.absorbed_probability_mass == 1.5
    assert stats.retained_probability_mass == 1 and stats.removed_probability_mass == 0


def test_score_and_terminal_depth_are_part_of_equivalence():
    def step(state, rng):
        if not state.data:
            return [
                (1, State({"end": 1}, terminal=True), "early"),
                (1, State({"continue": 1}), "continue"),
            ]
        return [
            (1, State({"end": 1}, terminal=True), "late"),
            (1, State({"end": 1}, score=2, terminal=True), "scored"),
        ]

    result = Simulation(step, SimulationConfig(deduplication="state_markov")).run(
        State()
    )
    assert len(result.branches) == 3
    assert result.statistics.branches_merged == 0
    assert sorted(b.steps for b in result.branches) == [1, 2, 2]


def test_equivalence_is_fingerprint_based_not_mapping_order():
    result = Simulation(
        lambda s, r: [
            (1, State({"a": 1, "b": 2}).stop(), "a"),
            (1, State({"b": 2, "a": 1}).stop(), "b"),
        ],
        SimulationConfig(deduplication="state_markov"),
    ).run(State())
    assert len(result.branches) == 1 and result.best().probability == 1


def test_invalid_fingerprint_data_is_only_rejected_when_merging():
    def step(s, r):
        return [(1, State({"opaque": object()}, terminal=True), "object")]

    assert Simulation(step).run(State()).best() is not None
    with pytest.raises(TypeError, match="unsupported"):
        Simulation(step, SimulationConfig(deduplication="state_markov")).run(State())


@pytest.mark.parametrize("value", [True, "state", "approximate", None])
def test_invalid_equivalence_contract(value):
    with pytest.raises(ValueError, match="deduplication"):
        SimulationConfig(deduplication=value)


@pytest.mark.parametrize(
    "strategy, count, mass",
    [
        (NoPruning(), 3, 1),
        (TopK(1), 1, 0.65),
        (TopK(0), 0, 0),
        (MinProbability(0.18), 2, 0.85),
    ],
)
def test_strategies_replace_config_limits_and_account_for_loss(strategy, count, mass):
    result = Simulation(
        choices,
        SimulationConfig(max_branches=1, min_probability=0.99),
        pruning=strategy,
    ).run(State())
    assert len(result.branches) == count
    assert result.statistics.retained_probability_mass == pytest.approx(mass)
    assert result.statistics.removed_probability_mass == pytest.approx(1 - mass)
    assert result.statistics.branches_pruned == 3 - count
    assert result.statistics.branches_retained == count


def test_no_pruning_still_obeys_execution_limits_and_ignores_zero_weights():
    result = Simulation(
        lambda s, r: [(0, s, "zero"), (1, s, "a"), (1, s, "b")],
        SimulationConfig(max_steps=2, max_branches=1),
        pruning=NoPruning(),
    ).run(State())
    assert result.termination_reasons == (Reason.MAX_STEPS,)
    assert result.statistics.transitions_generated == 9
    assert result.statistics.branches_generated == 6
    assert result.statistics.states_visited == 3 and len(result.branches) == 4


def test_custom_strategy_and_ordering():
    class Custom:
        def select(self, branches):
            assert isinstance(branches, tuple)
            return [1, 0]

    result = Simulation(choices, pruning=Custom()).run(State())
    assert [b.history for b in result.branches] == [("a",), ("b",)]
    assert result.termination_reasons == (Reason.CUSTOM_PRUNING, Reason.COMPLETED)
    assert result.statistics.removed_probability_mass == pytest.approx(0.65)


@pytest.mark.parametrize("indices", [[0, 0], [-1], [3], [True], ["0"]])
def test_invalid_strategy_outputs(indices):
    class Invalid:
        def select(self, branches):
            return indices

    with pytest.raises(ValueError, match="pruning strategy"):
        Simulation(choices, pruning=Invalid()).run(State())


@pytest.mark.parametrize("k", [-1, 1.5, True])
def test_invalid_top_k(k):
    with pytest.raises(ValueError):
        TopK(k)


@pytest.mark.parametrize("p", [-1, 2, float("nan"), float("inf")])
def test_invalid_threshold(p):
    with pytest.raises(ValueError):
        MinProbability(p)


def test_dedup_determinism_and_long_run_mass_accounting():
    class Walk:
        def step(self, state, rng):
            tick = state.data.get("tick", 0) + 1
            return [
                (0.4, State({"tick": tick, "x": state.data.get("x", 0) + 1}), "up"),
                (0.6, State({"tick": tick, "x": state.data.get("x", 0) - 1}), "down"),
            ]

    scenario = Scenario(
        Walk(),
        config=SimulationConfig(max_steps=20, seed=7, deduplication="state_markov"),
        pruning=TopK(5),
    )
    first, second = scenario.run(), scenario.run()
    assert first.branches == second.branches
    assert math.isclose(
        first.statistics.retained_probability_mass
        + first.statistics.removed_probability_mass,
        1,
        abs_tol=1e-12,
    )
    metrics = asdict(first.statistics)
    assert metrics["branches_merged"] > 0 and metrics["branches_pruned"] > 0
