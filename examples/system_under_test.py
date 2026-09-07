"""Exercise an ordinary standard-library function without an Agent wrapper."""

import statistics

from forka import Scenario, State, SystemModel


def to_samples(state):
    return state.data["samples"]


def from_mean(state, value):
    return state.with_data(mean=value).stop()


if __name__ == "__main__":
    model = SystemModel(statistics.mean, to_samples, from_mean)
    result = Scenario(model, State({"samples": [1, 2, 3, 4]})).run()
    assert result.best().state.data["mean"] == 2.5
    print("System output:", result.best().state.data)
