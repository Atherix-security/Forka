from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    value: Any
    created_at: float


class MemoryCache:
    def __init__(self):
        self._data: dict[str, CacheEntry] = {}

    @staticmethod
    def key(value: Any) -> str:
        raw = json.dumps(value, sort_keys=True, default=str).encode()
        return hashlib.sha256(raw).hexdigest()

    def get(self, key: str, ttl: float | None = None) -> Any | None:
        if ttl is not None and (not math.isfinite(ttl) or ttl < 0):
            raise ValueError("ttl must be finite and nonnegative")
        entry = self._data.get(key)
        if entry is None or (
            ttl is not None and time.monotonic() - entry.created_at >= ttl
        ):
            return None
        return entry.value

    def set(self, key: str, value: Any) -> None:
        self._data[key] = CacheEntry(value, time.monotonic())
