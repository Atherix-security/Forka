"""Abstract attacker/defender risk game: two agents, no network or LLM calls."""

from forka import (
    Action,
    Agent,
    Model,
    Observation,
    Scenario,
    SimulationConfig,
    State,
)


class RiskEnvironment:
    """Shared rules; every branch carries its own exposure and turn counter."""

    def observe(self, state, actor):
        return Observation({"exposure": state.data["exposure"], "actor": actor})

    def apply(self, state, actor, action, rng):
        change = action.data["amount"] * (1 if actor == "attacker" else -1)
        turn = state.data["turn"] + 1
        next_state = state.with_data(
            exposure=max(0, state.data["exposure"] + change), turn=turn
        )
        return next_state.stop() if turn >= 4 else next_state


def attack_policy(observation, rng):
    return [
        Action("low_pressure", {"amount": 1}, probability=0.7),
        Action("high_pressure", {"amount": 3}, probability=0.3),
    ]


def defend_policy(observation, rng):
    # A rule-based policy responding to its own view of the shared world.
    strong_weight = 0.8 if observation.data["exposure"] >= 3 else 0.4
    return [
        Action("basic_response", {"amount": 1}, probability=1 - strong_weight),
        Action("strong_response", {"amount": 2}, probability=strong_weight),
    ]


class AlternatingAgents(Model):
    def __init__(self, attacker, defender):
        self.attacker = attacker
        self.defender = defender

    def step(self, state, rng):
        agent = self.attacker if state.data["turn"] % 2 == 0 else self.defender
        return agent.step(state, rng)


def build_scenario(seed=7):
    environment = RiskEnvironment()
    attacker = Agent("attacker", environment, attack_policy)
    defender = Agent("defender", environment, defend_policy)
    return Scenario(
        model=AlternatingAgents(attacker, defender),
        initial_state=State({"exposure": 0, "turn": 0}),
        config=SimulationConfig(max_steps=4, max_branches=32, seed=seed),
        evaluator=lambda state: -state.data["exposure"],
    )


if __name__ == "__main__":
    result = build_scenario().run()
    print("Lowest exposure:", result.best())
    print("Termination:", [reason.value for reason in result.termination_reasons])
    print("Statistics:", result.statistics)
