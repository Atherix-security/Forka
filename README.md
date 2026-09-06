# Forka

**Explore what happens before it happens.**

Forka is an open-source, compute-efficient scenario simulation framework for
researchers, scientists, engineers, businesses, hobbyists, and AI developers.
Use it to orchestrate scientific, statistical, rule-based or deterministic models,
including intelligent agents. Agents are first-class and optional. The core is
CPU-first: no GPU, paid API or runtime dependency is required.

> **Status:** v0.2 foundation in development (`0.2.0.dev0`). The v0.1 kernel API
> remains supported. APIs may change before v1.0.

## Why Forka?

Exploring alternative outcomes should work on a laptop. Forka supplies branching,
pruning, reproducible execution and result accounting around your domain model.
It orchestrates scientific libraries rather than replacing their numerical methods.
Agent models can use rules or optional inference providers when appropriate.

- CPU-first simulation kernel
- deterministic and probabilistic branching
- branch and runtime limits
- reproducible seeded runs
- provider-agnostic intelligence routing
- compute/cost budget primitives
- caching primitives to avoid repeated work
- zero runtime dependencies for the core package
- generic Model and Scenario interfaces, with optional Agent and Environment composition
- evaluator scoring, deterministic state fingerprints and structured result metrics

## Quick start

```bash
git clone https://github.com/Atherix-security/Forka.git
cd Forka
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1
pip install -e .
python examples/coin_branch.py
python examples/population_growth.py
python examples/attacker_defender.py
```

## Scientific model: no agents required

`Model` is a small protocol. Any object with `step(state, rng)` can be used without
inheriting from a Forka class. A `Scenario` binds that model to state and run limits.

```python
from forka import Model, Scenario, SimulationConfig, State, Transition


class Population(Model):
    def step(self, state, rng):
        year = state.data["year"] + 1
        for weight, rate in [(0.4, 0.05), (0.6, 0.12)]:
            yield Transition(
                weight,
                State(
                    {"population": state.data["population"] * (1 + rate), "year": year},
                    terminal=year >= 3,
                ),
                f"growth:{rate}",
            )


result = Scenario(
    Population(),
    State({"population": 100.0, "year": 0}),
    SimulationConfig(seed=7),
    evaluator=lambda state: state.data["population"],
).run()
print(result.statistics)
print(result.termination_reasons)
print(result.best().state.fingerprint())
```

See [population_growth.py](examples/population_growth.py) for seeded variation
alongside explicitly branched weather uncertainty. It is an illustrative model,
not a calibrated scientific forecast.

## Agents: models with policies and environment views

An `Agent` implements the same Model interface using a policy callable and an
Environment. Each policy returns weighted `Action` alternatives; the environment
maps each action into a successor state. `Observation` lets the environment
control which data is visible to a policy.

```python
from forka import Action, Agent, Observation, Scenario, State


class CounterWorld:
    def observe(self, state, actor):
        return Observation({"value": state.data["value"]})

    def apply(self, state, actor, action, rng):
        return state.with_data(value=state.data["value"] + action.data["delta"]).stop()


def policy(observation, rng):
    return [Action("increment", {"delta": 1}, 0.8), Action("hold", {"delta": 0}, 0.2)]


agent = Agent("counter", CounterWorld(), policy)
result = Scenario(agent, State({"value": 0})).run()
print(result.branches)
```

[attacker_defender.py](examples/attacker_defender.py) composes two agents sharing
one environment into an alternating-turn Model, branching over both policies.
All world state and turn information live in branch State values; neither example
uses an LLM or external service. Models can orchestrate other models directly.

## Existing v0.1 API

```python
from forka import Simulation, SimulationConfig, State, Transition


def step(state, rng):
    value = state.data.get("value", 0)
    return [
        Transition(0.6, state.with_data(value=value + 1), "advance"),
        Transition(0.4, state.with_data(value=value - 1), "retreat"),
    ]


result = Simulation(step, SimulationConfig(max_steps=8, max_branches=64, seed=7)).run(
    State({"value": 0})
)
print(result.best())
```

## Architecture and adapters

1. **Simulation kernel**: state, transitions, branches, pruning, scoring.
2. **Scenario runtime**: generic models, optional agents, environments and evaluators.
3. **Intelligence router**: deterministic logic, local models, cloud providers.
4. **Knowledge layer**: research, retrieval, caching, provenance.
5. **Compute scheduler**: adapt fidelity to hardware, time, and monetary budgets.

The guiding rule is simple: **more compute should increase scale and fidelity, not be required to use Forka.**

See [ROADMAP.md](ROADMAP.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
The [v0.2 architecture proposal](docs/architecture-v0.2.md) defines the boundaries,
metric meanings and future host considerations. The [API notes](docs/api-v0.2.md)
cover adapters, compatibility and technical debt.

## License

Apache-2.0. See [LICENSE](LICENSE).

## v0.1 behavior and limits

The kernel enumerates all returned transitions; the seeded `random.Random` passed
into your step function controls any randomness you choose to use. Return finite,
nonnegative weights with a positive total; Forka normalizes them per parent.
Zero-weight transitions are omitted. Pruning retains the highest-probability
branches (score breaks ties), without renormalizing retained probability mass.
`truncated` is true when positive-probability paths are pruned or execution stops
with unfinished branches, including the step limit. `steps_executed` counts
expansion rounds; individual branches track their own steps.

Runtime limits are cooperative and checked between step calls. A running step
function cannot be interrupted, so one call may exceed the time budget. State
helpers copy the top-level data mapping only; treat nested values as immutable
or copy them yourself when creating branches.

`ComputeBudget` tracks model calls and cost when explicitly used by an adapter.
`MemoryCache` is an in-memory cache with optional TTL on reads; cached values
are returned by reference, and `None` also represents a cache miss. Cache keys
are intended for JSON-compatible values. The router accepts callables and
objects with a `complete` method. It uses the preferred registered provider or
falls back to the first registered provider. Request complexity, web, and cost
fields are metadata for future routing policies; v0.1 does not enforce them.
Neither budgets nor caching are automatically wired into the router or kernel.

## Validation

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python examples/coin_branch.py
python examples/population_growth.py
python examples/attacker_defender.py
python -m build
```
