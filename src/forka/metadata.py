"""Allowlisted reproducibility metadata; never inspect model data or environment."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ._version import __version__
from .pruning import MinProbability, NoPruning, TopK


def qualified_name(value: Any) -> str:
    cls = value if hasattr(value, "__qualname__") else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


@lru_cache(maxsize=1)
def _source_snapshot() -> tuple[str | None, bool | None]:
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists() or not (root / "pyproject.toml").is_file():
        return None, None
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout.strip()
        dirty = (
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            ).stdout
            != ""
        )
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def pruning_record(strategy) -> dict[str, Any]:
    if strategy is None:
        return {"name": "config_defaults", "replaces_config_limits": False}
    record = {"name": qualified_name(strategy), "replaces_config_limits": True}
    if isinstance(strategy, MinProbability):
        record["threshold"] = strategy.threshold
    elif isinstance(strategy, TopK):
        record["k"] = strategy.k
    elif not isinstance(strategy, NoPruning):
        record["parameters_recorded"] = False
    return record


@dataclass(frozen=True)
class ReproducibilityRecord:
    forka_version: str
    git_commit: str | None
    git_dirty: bool | None
    seed: int | None
    simulation_config: dict[str, Any]
    experiment_config: dict[str, Any] | None
    worker_count: int
    model_identifier: str
    scenario_identifier: str | None
    evaluator_identifier: str | None
    python_version: str
    platform: str
    deterministic_settings: dict[str, Any]


def capture_record(
    config,
    model,
    *,
    pruning=None,
    evaluator=None,
    scenario_identifier=None,
    experiment_config=None,
    workers=1,
):
    commit, dirty = _source_snapshot()
    return ReproducibilityRecord(
        forka_version=__version__,
        git_commit=commit,
        git_dirty=dirty,
        seed=config.seed,
        simulation_config=asdict(config),
        experiment_config=experiment_config,
        worker_count=workers,
        model_identifier=qualified_name(model),
        scenario_identifier=scenario_identifier,
        evaluator_identifier=None if evaluator is None else qualified_name(evaluator),
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.release()} {platform.machine()}",
        deterministic_settings={
            "seeded": config.seed is not None,
            "rng": "random.Random; serial traversal order",
            "deduplication": config.deduplication,
            "runtime_limit_may_change_results": config.max_runtime_seconds is not None,
            "pruning": pruning_record(pruning),
            "source_snapshot": "cached once per process; unavailable for wheels",
        },
    )
