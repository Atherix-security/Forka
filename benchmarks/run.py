"""Reproducible measurements, with raw repetitions and no speedup assertions."""

import argparse
import json
import math
import os
import platform
import statistics
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path

from forka import (
    Experiment,
    MinProbability,
    NoPruning,
    Scenario,
    SimulationConfig,
    State,
    TopK,
)

from .workloads import CpuModel, LatticeModel


def distribution(result):
    grouped = {}
    for branch in result.branches:
        key = branch.state.fingerprint()
        grouped[key] = grouped.get(key, 0.0) + branch.probability
    return grouped


def measure(scenario, repeats):
    scenario.run()  # Warm imports and cached source metadata, outside timings.
    seconds = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = scenario.run()
        seconds.append(time.perf_counter() - started)
    stats = result.statistics
    assert math.isclose(
        stats.retained_probability_mass + stats.removed_probability_mass,
        1,
        abs_tol=1e-10,
    )
    # Measure memory separately so tracemalloc does not distort reported speed.
    tracemalloc.start()
    try:
        measured = scenario.run()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert distribution(measured) == distribution(result)
    median = statistics.median(seconds)
    return result, {
        "seconds": seconds,
        "median_seconds": median,
        "transitions_per_second": stats.transitions_generated / median if median else 0,
        "peak_python_bytes_separate_run": peak,
        "statistics": asdict(stats),
        "reproducibility": asdict(result.reproducibility),
    }


def run_benchmarks(*, depth=7, repeats=5, runs=128, work=10000):
    if any(type(n) is not int or n < 1 for n in (depth, repeats, runs, work)):
        raise ValueError("benchmark parameters must be positive integers")
    branch_cases = {}
    full_distributions = {}
    for divergent in (False, True):
        strategies = [("full", NoPruning())]
        if not divergent:
            strategies += [("top64", TopK(64)), ("threshold", MinProbability(0.0001))]
        for label, pruning in strategies:
            for mode in ("none", "state_markov"):
                name = f"{'divergent' if divergent else 'convergent'}_{label}_{mode}"
                scenario = Scenario(
                    LatticeModel(depth, divergent),
                    State({"tick": 0, "x": 0}),
                    SimulationConfig(max_steps=depth, seed=42, deduplication=mode),
                    pruning=pruning,
                    identifier=name,
                )
                result, row = measure(scenario, repeats)
                row["represented_paths"] = sum(b.path_count for b in result.branches)
                branch_cases[name] = row
                if label == "full":
                    full_distributions[(divergent, mode)] = distribution(result)
                    assert row["represented_paths"] == 4**depth
    for divergent in (False, True):
        plain = full_distributions[(divergent, "none")]
        merged = full_distributions[(divergent, "state_markov")]
        assert plain.keys() == merged.keys()
        assert all(
            math.isclose(value, merged[key], rel_tol=1e-10)
            for key, value in plain.items()
        )

    experiment_cases = {}
    reference = None
    for workers in (1, 2):
        experiment = Experiment(
            Scenario(CpuModel(work)), runs=runs, seed=42, workers=workers
        )
        experiment.run()  # Pool startup remains included in every timed run.
        samples = [experiment.run() for _ in range(repeats)]
        semantic = [run.branches for run in samples[-1].results]
        if reference is None:
            reference = semantic
        assert semantic == reference
        durations = [sample.statistics.runtime_seconds for sample in samples]
        median = statistics.median(durations)
        experiment_cases[str(workers)] = {
            "seconds": durations,
            "median_seconds": median,
            "simulations_per_second": runs / median if median else 0,
            "statistics": asdict(samples[-1].statistics),
            "reproducibility": asdict(samples[-1].reproducibility),
        }
    return {
        "hardware": {
            "processor": platform.processor() or "unavailable",
            "logical_cpus": os.cpu_count(),
            "architecture": platform.machine(),
        },
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "configuration": {
            "depth": depth,
            "repeats": repeats,
            "runs": runs,
            "work": work,
        },
        "method": "one warmup; raw timed repetitions; separate tracemalloc run; all cases reported",
        "branch_cases": branch_cases,
        "experiment_cases": experiment_cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--runs", type=int, default=128)
    parser.add_argument("--work", type=int, default=10000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = dict(
        depth=args.depth, repeats=args.repeats, runs=args.runs, work=args.work
    )
    if args.smoke:
        config = dict(depth=3, repeats=1, runs=4, work=20)
    report = run_benchmarks(**config)
    # Record the effective parameters; avoid storing arbitrary argv or local paths.
    report["command"] = "python -m benchmarks.run " + " ".join(
        f"--{key} {value}" for key, value in config.items()
    )
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Benchmark report written to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
