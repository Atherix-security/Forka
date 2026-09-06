from unittest.mock import patch

import pytest

from forka import (
    Branch,
    Simulation,
    SimulationConfig,
    SimulationResult,
    SimulationStatistics,
    State,
)
from forka import (
    TerminationReason as Reason,
)


def test_two_level_tree_counts():
    def step(state, rng):
        level = state.data.get("level", 0) + 1
        return [
            (1, State({"level": level}, terminal=level == 2), label)
            for label in ("a", "b")
        ]

    result = Simulation(step).run(State())
    assert result.statistics == SimulationStatistics(
        6, 0, 4, 1.0, result.elapsed_seconds
    )
    assert result.termination_reasons == (Reason.COMPLETED,)
    assert result.steps_executed == 2


def test_cap_and_threshold_pruning_counts_without_renormalization():
    result = Simulation(
        lambda s, r: [(w, s.stop(), str(w)) for w in [0, 0.1, 0.2, 0.3, 0.4]],
        SimulationConfig(min_probability=0.15, max_branches=2),
    ).run(State())
    assert result.statistics.branches_explored == 4
    assert result.statistics.branches_pruned == 2
    assert result.statistics.peak_active_branches == 2
    assert result.statistics.retained_probability_mass == pytest.approx(0.7)
    assert result.termination_reasons == (
        Reason.MIN_PROBABILITY,
        Reason.MAX_BRANCHES,
        Reason.COMPLETED,
    )
    assert result.truncated


def test_repeated_pruning_reason_appears_once():
    result = Simulation(
        lambda s, r: [(1, s, "a"), (1, s, "b")],
        SimulationConfig(max_steps=3, max_branches=1),
    ).run(State())
    assert result.termination_reasons == (Reason.MAX_BRANCHES, Reason.MAX_STEPS)
    assert result.statistics.branches_explored == 6
    assert result.statistics.branches_pruned == 3
    assert result.statistics.retained_probability_mass == 0.125


def test_all_branches_pruned_and_best_empty():
    result = Simulation(
        lambda s, r: [(1, s, "a"), (1, s, "b")],
        SimulationConfig(min_probability=0.6),
    ).run(State())
    assert result.best() is None
    assert result.statistics == SimulationStatistics(2, 2, 1, 0, result.elapsed_seconds)
    assert result.termination_reasons == (Reason.MIN_PROBABILITY, Reason.NO_BRANCHES)


@pytest.mark.parametrize(
    "terminal, reason, truncated",
    [(True, Reason.COMPLETED, False), (False, Reason.MAX_STEPS, True)],
)
def test_zero_steps(terminal, reason, truncated):
    result = Simulation(lambda s, r: [], SimulationConfig(max_steps=0)).run(
        State(terminal=terminal)
    )
    assert result.statistics == SimulationStatistics(0, 0, 1, 1, result.elapsed_seconds)
    assert result.termination_reasons == (reason,)
    assert result.truncated is truncated


def test_empty_step_does_not_count_a_generated_branch():
    result = Simulation(lambda s, r: []).run(State())
    assert result.statistics == SimulationStatistics(0, 0, 1, 1, result.elapsed_seconds)
    assert result.branches[0].steps == 0  # Preserve v0.1's step accounting.
    assert result.steps_executed == 1
    assert result.termination_reasons == (Reason.COMPLETED,)


def test_terminal_branches_are_carried_without_double_counting():
    def step(state, rng):
        if not state.data:
            return [(1, State(terminal=True), "done"), (1, State({"x": 1}), "continue")]
        return [(1, state.stop(), "done")]

    result = Simulation(step).run(State())
    assert result.statistics == SimulationStatistics(3, 0, 2, 1, result.elapsed_seconds)
    assert sorted(b.steps for b in result.branches) == [1, 2]


def test_runtime_expiry_mid_frontier_retains_unprocessed_branches():
    now = [0.0]

    def step(state, rng):
        if not state.data:
            return [(1, State({"x": i}), str(i)) for i in range(3)]
        now[0] = 2
        return [(1, state.stop(), "done")]

    with patch("forka.core.time.monotonic", side_effect=lambda: now[0]):
        result = Simulation(step, SimulationConfig(max_runtime_seconds=1)).run(State())
    assert result.termination_reasons == (Reason.MAX_RUNTIME,)
    assert result.statistics == SimulationStatistics(4, 0, 3, 1, 2)
    assert result.truncated
    assert sorted(b.steps for b in result.branches) == [1, 1, 2]


def test_zero_runtime_does_not_call_model():
    def fail(s, r):
        raise AssertionError("must not run")

    result = Simulation(fail, SimulationConfig(max_runtime_seconds=0)).run(State())
    assert result.termination_reasons == (Reason.MAX_RUNTIME,)
    assert result.statistics.branches_explored == 0


def test_underflow_is_recorded_as_probability_pruning():
    result = Simulation(
        lambda s, r: [(1e308, s.stop(), "large"), (1e-308, s.stop(), "tiny")],
        SimulationConfig(min_probability=0),
    ).run(State())
    assert result.statistics.branches_explored == 2
    assert result.statistics.branches_pruned == 1
    assert result.termination_reasons == (Reason.MIN_PROBABILITY, Reason.COMPLETED)


def test_legacy_result_positional_constructor_and_enum_json():
    import json

    result = SimulationResult([Branch(State())], 0.5, 1, True)
    assert result.truncated and result.best() is result.branches[0]
    assert result.termination_reasons == ()
    assert result.statistics == SimulationStatistics()
    assert json.dumps([Reason.MAX_STEPS]) == '["max_steps"]'
