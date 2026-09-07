"""Independent seeded runs with optional local spawn workers."""

from __future__ import annotations

import copy
import hashlib
import multiprocessing
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Iterable

from .core import SimulationResult
from .metadata import ReproducibilityRecord, capture_record, qualified_name
from .runtime import Scenario


def run_seed(seed: int, index: int) -> int:
    """Stable 64-bit seed for an index, independent of worker/scheduling order."""
    if type(seed) is not int or type(index) is not int or index < 0:
        raise ValueError("seed must be an integer and index a nonnegative integer")
    payload = f"forka-experiment-v1:{seed}:{index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _serial_run(scenario: Scenario, seed: int) -> SimulationResult:
    independent = copy.deepcopy(scenario)
    return replace(independent, config=replace(independent.config, seed=seed)).run()


_template: bytes | None = None


def _initialize_worker(template: bytes):
    global _template
    _template = template


def _worker_run(seed: int) -> SimulationResult:
    # Only bytes produced locally from the caller's trusted scenario are loaded.
    independent = pickle.loads(_template)
    return replace(independent, config=replace(independent.config, seed=seed)).run()


@dataclass(frozen=True)
class ExperimentStatistics:
    runs_completed: int
    worker_count: int
    runtime_seconds: float
    states_visited: int
    transitions_generated: int
    branches_generated: int
    branches_retained: int
    branches_merged: int
    states_deduplicated: int
    branches_pruned: int
    peak_active_branches: int
    mean_retained_probability_mass: float | None
    mean_removed_probability_mass: float | None

    @property
    def simulations_per_second(self) -> float:
        return (
            self.runs_completed / self.runtime_seconds
            if self.runtime_seconds > 0
            else 0.0
        )

    @property
    def transitions_per_second(self) -> float:
        return (
            self.transitions_generated / self.runtime_seconds
            if self.runtime_seconds > 0
            else 0.0
        )


@dataclass(frozen=True)
class ExperimentResult:
    seeds: tuple[int, ...]
    results: tuple[SimulationResult, ...]
    statistics: ExperimentStatistics
    reproducibility: ReproducibilityRecord


@dataclass(frozen=True)
class Experiment:
    scenario: Scenario
    runs: int = 1
    seed: int = 0
    workers: int = 1
    keep_results: bool = True

    def __post_init__(self):
        if type(self.runs) is not int or self.runs < 0:
            raise ValueError("runs must be a nonnegative integer")
        if type(self.workers) is not int or self.workers < 1:
            raise ValueError("workers must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("experiment seed must be an integer")
        if type(self.keep_results) is not bool:
            raise ValueError("keep_results must be a boolean")

    def run(self) -> ExperimentResult:
        seeds = tuple(run_seed(self.seed, index) for index in range(self.runs))
        effective_workers = min(self.workers, self.runs)
        record = capture_record(
            self.scenario.config,
            self.scenario.model,
            pruning=self.scenario.pruning,
            evaluator=self.scenario.evaluator,
            scenario_identifier=self.scenario.identifier
            or qualified_name(self.scenario.model),
            workers=effective_workers,
            experiment_config={
                "runs": self.runs,
                "seed": self.seed,
                "workers": self.workers,
                "effective_workers": effective_workers,
                "keep_results": self.keep_results,
                "seed_scheme": "forka-experiment-v1",
                "start_method": "spawn" if effective_workers > 1 else "serial",
            },
        )
        record = replace(record, seed=self.seed)
        started = time.perf_counter()
        results = []
        totals = dict(
            states_visited=0,
            transitions_generated=0,
            branches_generated=0,
            branches_retained=0,
            branches_merged=0,
            states_deduplicated=0,
            branches_pruned=0,
        )
        peak = 0
        retained = removed = 0.0
        completed = 0

        def consume(stream: Iterable[SimulationResult]):
            nonlocal peak, retained, removed, completed
            for result in stream:
                stats = result.statistics
                for name in totals:
                    totals[name] += getattr(stats, name)
                peak = max(peak, stats.peak_active_branches)
                retained += stats.retained_probability_mass
                removed += stats.removed_probability_mass
                completed += 1
                if self.keep_results:
                    results.append(result)

        if effective_workers > 1:
            try:
                template = pickle.dumps(self.scenario)
            except (TypeError, AttributeError, pickle.PickleError) as error:
                raise ValueError(
                    "parallel scenarios must be picklable; use importable classes/functions, not lambdas or local definitions"
                ) from error
            with ProcessPoolExecutor(
                max_workers=effective_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
                initargs=(template,),
            ) as pool:
                consume(pool.map(_worker_run, seeds))
        else:
            consume(_serial_run(self.scenario, seed) for seed in seeds)
        elapsed = time.perf_counter() - started
        statistics = ExperimentStatistics(
            runs_completed=completed,
            worker_count=effective_workers,
            runtime_seconds=elapsed,
            **totals,
            peak_active_branches=peak,
            mean_retained_probability_mass=retained / completed if completed else None,
            mean_removed_probability_mass=removed / completed if completed else None,
        )
        return ExperimentResult(seeds, tuple(results), statistics, record)
