"""Run with --workers 2 for a local spawn-process experiment."""

import argparse

from forka import Experiment, Model, Scenario, SimulationConfig, State, Transition


class DiceModel(Model):
    def step(self, state, rng):
        return [Transition(1, state.with_data(value=rng.randint(1, 6)).stop(), "roll")]


def build_experiment(workers=1):
    return Experiment(
        Scenario(DiceModel(), State(), SimulationConfig(max_steps=1)),
        runs=100,
        seed=42,
        workers=workers,
    )


if __name__ == "__main__":  # Required for spawn multiprocessing on Windows/POSIX.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    result = build_experiment(parser.parse_args().workers).run()
    mean = sum(run.best().state.data["value"] for run in result.results) / len(
        result.results
    )
    print(
        f"100 seeded rolls: mean={mean:.3g}, elapsed={result.statistics.runtime_seconds:.3g}s"
    )
    print("First three seeds:", result.seeds[:3])
    print("Effective CPU workers:", result.statistics.worker_count)
