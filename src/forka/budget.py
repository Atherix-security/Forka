import math
from dataclasses import dataclass


@dataclass
class ComputeBudget:
    max_model_calls: int | None = None
    max_cost: float | None = None
    model_calls: int = 0
    cost: float = 0.0

    def __post_init__(self):
        for name in ("max_model_calls", "model_calls"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("max_cost", "cost"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_model_calls is not None and self.model_calls > self.max_model_calls:
            raise ValueError("model_calls exceeds max_model_calls")
        if self.max_cost is not None and self.cost > self.max_cost:
            raise ValueError("cost exceeds max_cost")

    def can_spend(self, estimated_cost: float = 0.0) -> bool:
        if not math.isfinite(estimated_cost) or estimated_cost < 0:
            raise ValueError("estimated cost must be finite and nonnegative")
        calls_ok = (
            self.max_model_calls is None or self.model_calls < self.max_model_calls
        )
        cost_ok = self.max_cost is None or self.cost + estimated_cost <= self.max_cost
        return calls_ok and cost_ok

    def spend(self, cost: float = 0.0) -> None:
        if not self.can_spend(cost):
            raise RuntimeError("Compute budget exhausted")
        self.model_calls += 1
        self.cost += cost
