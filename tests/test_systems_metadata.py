import json
import statistics
from dataclasses import asdict

import pytest

from forka import CallableSystem, Scenario, SimulationConfig, State, SystemModel, TopK


def to_samples(state):
    return state.data["samples"]


def from_output(state, output):
    return state.with_data(result=output).stop()


class ObjectSystem:
    def run(self, value):
        return sum(value)


@pytest.mark.parametrize(
    "system, expected",
    [(statistics.mean, 2), (CallableSystem(sum), 6), (ObjectSystem(), 6)],
)
def test_external_functions_and_objects_do_not_require_agents(system, expected):
    model = SystemModel(system, to_samples, from_output)
    result = Scenario(model, State({"samples": [1, 2, 3]})).run()
    assert result.best().state.data["result"] == expected
    assert result.statistics.states_visited == 1


def test_external_errors_and_bad_adapters_propagate():
    def broken(value):
        raise RuntimeError("test system failed")

    with pytest.raises(RuntimeError, match="test system failed"):
        Scenario(
            SystemModel(broken, to_samples, from_output), State({"samples": []})
        ).run()
    with pytest.raises(TypeError, match="output_adapter"):
        Scenario(
            SystemModel(sum, to_samples, lambda s, o: o), State({"samples": [1]})
        ).run()


def test_reproducibility_allowlist_and_effective_pruning(monkeypatch):
    monkeypatch.setenv("FORKA_PRIVATE_TOKEN", "do-not-record-this-secret")
    model = SystemModel(statistics.mean, to_samples, from_output)
    result = Scenario(
        model,
        State({"samples": [1, 2], "token": "private-input"}),
        SimulationConfig(seed=7),
        pruning=TopK(2),
        identifier="mean-check",
    ).run()
    record = asdict(result.reproducibility)
    encoded = json.dumps(record)
    assert "do-not-record-this-secret" not in encoded and "private-input" not in encoded
    assert "samples" not in encoded and "environment" not in record
    assert record["seed"] == 7 and record["simulation_config"]["seed"] == 7
    assert record["scenario_identifier"] == "mean-check"
    assert record["model_identifier"].endswith("SystemModel")
    assert record["python_version"] and record["platform"]
    assert record["deterministic_settings"]["pruning"]["k"] == 2
    assert record["deterministic_settings"]["pruning"]["replaces_config_limits"]


def test_unknown_custom_strategy_does_not_serialize_private_parameters():
    class Custom:
        token = "strategy-private-value"

        def select(self, branches):
            return range(len(branches))

    result = Scenario(
        SystemModel(sum, to_samples, from_output),
        State({"samples": [1]}),
        pruning=Custom(),
    ).run()
    pruning = result.reproducibility.deterministic_settings["pruning"]
    assert pruning["parameters_recorded"] is False
    assert "strategy-private-value" not in json.dumps(asdict(result.reproducibility))


@pytest.mark.parametrize("seed", [True, "secret string", 1.5])
def test_simulation_seed_respects_declared_public_type(seed):
    with pytest.raises(ValueError, match="seed"):
        SimulationConfig(seed=seed)
