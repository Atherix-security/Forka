from dataclasses import asdict

import pytest

from forka import Experiment, Scenario, SimulationConfig, run_seed


class RandomModel:
    def step(self, state, rng):
        return [(1, state.with_data(value=rng.random()).stop(), "sample")]


class StatefulModel:
    def __init__(self):
        self.calls = 0

    def step(self, state, rng):
        self.calls += 1
        return [
            (1, state.with_data(calls=self.calls, value=rng.random()).stop(), "sample")
        ]


class FailingModel:
    def step(self, state, rng):
        raise RuntimeError("model run failed")


def test_seeds_reproduce_individual_runs_and_prefixes():
    assert tuple(run_seed(42, i) for i in range(3)) == (
        6317397355638699482,
        3074948174631659021,
        9115352203832463961,
    )
    scenario = Scenario(RandomModel(), config=SimulationConfig(seed=999))
    experiment = Experiment(scenario, runs=8, seed=42)
    a, b = experiment.run(), experiment.run()
    assert a.seeds == b.seeds == tuple(run_seed(42, i) for i in range(8))
    assert len(set(a.seeds)) == 8
    assert [r.branches for r in a.results] == [r.branches for r in b.results]
    assert a.seeds[:3] == Experiment(scenario, runs=3, seed=42).run().seeds
    for seed, result in zip(a.seeds, a.results):
        replay = Scenario(RandomModel(), config=SimulationConfig(seed=seed)).run()
        assert replay.branches == result.branches
        assert result.reproducibility.seed == seed
    assert scenario.config.seed == 999
    unseeded_template = Experiment(Scenario(RandomModel()), runs=1, seed=42).run()
    assert unseeded_template.reproducibility.simulation_config["seed"] is None
    assert unseeded_template.reproducibility.seed == 42
    assert unseeded_template.reproducibility.deterministic_settings["seeded"] is True
    assert a.seeds != Experiment(scenario, runs=8, seed=43).run().seeds


def test_spawn_workers_match_serial_and_reset_model_instances():
    model = StatefulModel()
    scenario = Scenario(model)
    serial = Experiment(scenario, runs=8, seed=3).run()
    parallel = Experiment(scenario, runs=8, seed=3, workers=2).run()
    assert serial.seeds == parallel.seeds
    assert [r.branches for r in serial.results] == [
        r.branches for r in parallel.results
    ]
    assert all(
        r.best().state.data["calls"] == 1 for r in serial.results + parallel.results
    )
    assert model.calls == 0
    assert parallel.statistics.worker_count == 2
    assert parallel.reproducibility.experiment_config["start_method"] == "spawn"


def test_summary_only_results_and_aggregates():
    full = Experiment(Scenario(RandomModel()), runs=4).run()
    summary = Experiment(Scenario(RandomModel()), runs=4, keep_results=False).run()
    assert summary.results == () and summary.seeds == full.seeds
    stats = summary.statistics
    assert (
        stats.runs_completed
        == stats.states_visited
        == stats.transitions_generated
        == stats.branches_generated
        == 4
    )
    assert stats.branches_merged == stats.branches_pruned == 0
    assert (
        stats.mean_retained_probability_mass == 1
        and stats.mean_removed_probability_mass == 0
    )
    assert stats.peak_active_branches == 1
    assert stats.simulations_per_second > 0 and stats.transitions_per_second > 0
    for key, value in asdict(full.statistics).items():
        if key != "runtime_seconds":
            assert getattr(summary.statistics, key) == value


def test_zero_runs_are_empty_and_do_not_start_workers():
    result = Experiment(Scenario(RandomModel()), runs=0, workers=2).run()
    assert result.results == result.seeds == ()
    assert result.statistics.runs_completed == result.statistics.worker_count == 0
    assert result.statistics.mean_retained_probability_mass is None
    assert result.statistics.mean_removed_probability_mass is None
    assert result.statistics.simulations_per_second == 0


@pytest.mark.parametrize(
    "config",
    [
        {"runs": -1},
        {"runs": True},
        {"runs": 1.5},
        {"workers": 0},
        {"workers": False},
        {"workers": 1.5},
        {"seed": None},
        {"seed": True},
        {"keep_results": 1},
    ],
)
def test_malformed_experiment_config(config):
    with pytest.raises(ValueError):
        Experiment(Scenario(RandomModel()), **config)


@pytest.mark.parametrize("workers", [1, 2])
def test_run_failures_are_not_silently_successful(workers):
    with pytest.raises(RuntimeError, match="model run failed"):
        Experiment(Scenario(FailingModel()), runs=2, workers=workers).run()


def test_empty_simulation_results_can_be_aggregated():
    from forka import TopK

    result = Experiment(Scenario(RandomModel(), pruning=TopK(0)), runs=3).run()
    assert all(run.branches == [] for run in result.results)
    assert result.statistics.mean_retained_probability_mass == 0
    assert result.statistics.mean_removed_probability_mass == 1
    assert result.statistics.branches_retained == 0


def test_unpicklable_parallel_scenario_fails_clearly():
    scenario = Scenario(RandomModel(), evaluator=lambda state: 0)
    assert Experiment(scenario).run().statistics.runs_completed == 1
    with pytest.raises(ValueError, match="picklable"):
        Experiment(scenario, runs=2, workers=2).run()


@pytest.mark.parametrize("seed, index", [(True, 0), (1, -1), (1, 0.5)])
def test_invalid_seed_derivation_inputs(seed, index):
    with pytest.raises(ValueError):
        run_seed(seed, index)
