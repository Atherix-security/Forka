"""Portable, opt-in identity for state snapshots (not execution checkpoints)."""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import State


def _canonical(value: Any, ancestors: set[int]) -> Any:
    kind = type(value)
    if value is None:
        return ["null"]
    if kind is bool:
        return ["bool", value]
    if kind is int:
        return ["int", str(value)]
    if kind is float:
        if not math.isfinite(value):
            raise ValueError("state fingerprint requires finite numbers")
        return ["float", value.hex()]
    if kind is str:
        return ["str", value]
    if kind not in (dict, list):
        raise TypeError(f"unsupported fingerprint value: {kind.__name__}")
    identity = id(value)
    if identity in ancestors:
        raise ValueError("state fingerprint does not support cycles")
    ancestors.add(identity)
    try:
        if kind is list:
            return ["list", [_canonical(item, ancestors) for item in value]]
        if any(type(key) is not str for key in value):
            raise TypeError("state fingerprint requires string dictionary keys")
        return [
            "dict",
            [[key, _canonical(value[key], ancestors)] for key in sorted(value)],
        ]
    finally:
        ancestors.remove(identity)


def state_fingerprint(state: State) -> str:
    """Hash data, score and terminal flag with canonical format ``forka-state-v1``.

    Scalar types and list order are significant; dictionary insertion order is
    not. Convert external objects through an adapter before calling this function.
    """
    payload = [
        "forka-state-v1",
        _canonical(state.data, set()),
        _canonical(state.score, set()),
        _canonical(state.terminal, set()),
    ]
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
