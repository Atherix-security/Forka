from forka import Simulation, SimulationConfig, State, Transition


def step(state, rng):
    value = state.data.get("value", 0)
    n = state.data.get("turn", 0) + 1
    done = n >= 6
    return [
        Transition(
            0.6,
            State({"value": value + 1, "turn": n}, score=value + 1, terminal=done),
            "advance",
        ),
        Transition(
            0.4,
            State({"value": value - 1, "turn": n}, score=value - 1, terminal=done),
            "retreat",
        ),
    ]


result = Simulation(step, SimulationConfig(max_steps=6, max_branches=32, seed=7)).run(
    State({"value": 0, "turn": 0})
)
print(
    f"branches={len(result.branches)} elapsed={result.elapsed_seconds:.4f}s truncated={result.truncated}"
)
print("best:", result.best())
