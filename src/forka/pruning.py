"""Subset-selection strategies. No strategy may mutate its input branches."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Protocol, Sequence

if TYPE_CHECKING:
    from .core import Branch


class PruningStrategy(Protocol):
    def select(self, branches: Sequence[Branch]) -> Iterable[int]: ...


@dataclass(frozen=True)
class NoPruning:
    def select(self, branches: Sequence[Branch]) -> Iterable[int]:
        return range(len(branches))


@dataclass(frozen=True)
class MinProbability:
    threshold: float

    def __post_init__(self):
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be finite and between 0 and 1")

    def select(self, branches: Sequence[Branch]) -> Iterable[int]:
        return (
            i
            for i, branch in enumerate(branches)
            if branch.probability >= self.threshold
        )


@dataclass(frozen=True)
class TopK:
    k: int

    def __post_init__(self):
        if type(self.k) is not int or self.k < 0:
            raise ValueError("k must be a nonnegative integer")

    def select(self, branches: Sequence[Branch]) -> Iterable[int]:
        if self.k >= len(branches):
            return range(len(branches))
        return heapq.nlargest(
            self.k,
            range(len(branches)),
            key=lambda i: (branches[i].probability, branches[i].state.score, -i),
        )


def select_branches(branches: list[Branch], strategy: PruningStrategy):
    """Validate a strategy's subset and preserve original tie/traversal order."""
    if type(strategy) is NoPruning or (
        type(strategy) is TopK and strategy.k >= len(branches)
    ):
        return branches, 0, 0.0
    selected = list(strategy.select(tuple(branches)))
    if any(type(i) is not int or i < 0 or i >= len(branches) for i in selected):
        raise ValueError("pruning strategy returned an invalid branch index")
    keep = set(selected)
    if len(keep) != len(selected):
        raise ValueError("pruning strategy returned duplicate branch indices")
    removed_mass = math.fsum(
        b.probability for i, b in enumerate(branches) if i not in keep
    )
    return (
        [b for i, b in enumerate(branches) if i in keep],
        len(branches) - len(keep),
        removed_mass,
    )
