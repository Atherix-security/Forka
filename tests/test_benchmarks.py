import runpy
from pathlib import Path

from forka import NoPruning, Scenario, SimulationConfig, State


def test_benchmark_convergence_preserves_exact_distribution_and_paths():
    models = runpy.run_path(
        str(Path(__file__).parents[1] / "benchmarks" / "workloads.py")
    )
    results = []
    distributions = []
    for mode in ("none", "state_markov"):
        result = Scenario(
            models["LatticeModel"](depth=3),
            State({"tick": 0, "x": 0}),
            SimulationConfig(seed=42, deduplication=mode),
            pruning=NoPruning(),
        ).run()
        results.append(result)
        mass = {}
        for branch in result.branches:
            x = branch.state.data["x"]
            mass[x] = mass.get(x, 0) + branch.probability
        distributions.append(mass)
        assert sum(b.path_count for b in result.branches) == 64
    assert distributions[0] == distributions[1]
    assert results[0].statistics.branches_generated == 84
    assert results[1].statistics.branches_generated == 36
    assert results[1].statistics.branches_merged == 21
    assert len(results[0].branches) == 64 and len(results[1].branches) == 7


def test_benchmark_divergence_does_not_merge_distinct_paths():
    models = runpy.run_path(
        str(Path(__file__).parents[1] / "benchmarks" / "workloads.py")
    )
    result = Scenario(
        models["LatticeModel"](depth=3, divergent=True),
        State({"tick": 0, "x": 0}),
        SimulationConfig(seed=42, deduplication="state_markov"),
        pruning=NoPruning(),
    ).run()
    assert len(result.branches) == 64 and result.statistics.branches_merged == 0
    assert result.statistics.retained_probability_mass == 1
