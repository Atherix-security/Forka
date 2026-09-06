from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


class Provider(Protocol):
    def complete(self, prompt: str, **kwargs: Any) -> str: ...


@dataclass(frozen=True)
class RouteRequest:
    prompt: str
    complexity: str = "simple"
    requires_web: bool = False
    max_cost: float | None = None


class IntelligenceRouter:
    """Small, provider-agnostic router. Cloud/local adapters can register by name."""

    def __init__(self):
        self._providers: dict[str, Provider | Callable[..., str]] = {}

    def register(self, name: str, provider: Provider | Callable[..., str]) -> None:
        self._providers[name] = provider

    def route(self, request: RouteRequest, preferred: str | None = None) -> str:
        if not self._providers:
            raise RuntimeError("No intelligence providers are registered")
        name = (
            preferred if preferred in self._providers else next(iter(self._providers))
        )
        provider = self._providers[name]
        if hasattr(provider, "complete"):
            return provider.complete(request.prompt)  # type: ignore[attr-defined]
        return provider(request.prompt)  # type: ignore[operator]
