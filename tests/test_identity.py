import os
import subprocess
import sys

import pytest

from forka import State, state_fingerprint


def test_mapping_order_and_nested_identity():
    first = State({"b": [None, True, {"y": 2, "x": "hello"}], "a": 1.25})
    second = State({"a": 1.25, "b": [None, True, {"x": "hello", "y": 2}]})
    assert first.fingerprint() == state_fingerprint(second)
    assert len(first.fingerprint()) == 64
    assert first.fingerprint() == first.fingerprint()


def test_types_list_order_score_and_terminal_are_significant():
    states = [State({"x": x}) for x in [True, 1, 1.0, "1", None, [1, 2], [2, 1]]]
    states += [State({"x": 1}, score=1), State({"x": 1}, terminal=True)]
    assert len({state.fingerprint() for state in states}) == len(states)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -float("inf")])
def test_nonfinite_data_and_scores_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        State({"nested": [value]}).fingerprint()
    with pytest.raises(ValueError, match="finite"):
        State(score=value).fingerprint()


@pytest.mark.parametrize("value", [object(), {1, 2}, (1, 2), b"bytes", {1: "key"}])
def test_unsupported_objects_and_keys_rejected(value):
    with pytest.raises(TypeError):
        State({"value": value}).fingerprint()


def test_cycles_rejected_but_shared_values_are_allowed():
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="cycles"):
        State({"cycle": cycle}).fingerprint()
    shared = [1, 2]
    assert (
        State({"a": shared, "b": shared}).fingerprint()
        == State({"b": [1, 2], "a": [1, 2]}).fingerprint()
    )


def test_fingerprint_is_not_cached_on_mutable_v01_state_data():
    state = State({"x": 1})
    original = state.fingerprint()
    state.data["x"] = 2
    assert state.fingerprint() != original


def test_stable_in_separate_processes_with_different_python_hash_seeds():
    code = "from forka import State; print(State({k: [1, 'text'] for k in {'a','b','c'}}).fingerprint())"
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        ).strip()
        for seed in ("1", "99")
    ]
    assert (
        outputs[0]
        == outputs[1]
        == State({"a": [1, "text"], "b": [1, "text"], "c": [1, "text"]}).fingerprint()
    )
