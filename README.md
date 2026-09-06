# Forka

**Explore what happens before it happens.**

Forka is an open-source, compute-efficient simulation engine for AI agents, branching scenarios, and decision testing. The core is CPU-first: no GPU or API key is required to run useful simulations.

> **Status:** early v0.1 foundation. APIs may change before v1.0.

## Why Forka?

AI simulations can become expensive when every agent action requires a large-model call. Forka is designed to keep ordinary simulation work cheap and reserve AI inference for decisions that actually need it.

- CPU-first simulation kernel
- deterministic and probabilistic branching
- branch and runtime limits
- reproducible seeded runs
- provider-agnostic intelligence routing
- compute/cost budget primitives
- caching primitives to avoid repeated work
- zero runtime dependencies for the core package

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
```

## Example

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

## Architecture direction

1. **Simulation kernel**: state, transitions, branches, pruning, scoring.
2. **Agent runtime**: actors, policies, goals, memory.
3. **Intelligence router**: deterministic logic, local models, cloud providers.
4. **Knowledge layer**: research, retrieval, caching, provenance.
5. **Compute scheduler**: adapt fidelity to hardware, time, and monetary budgets.

The guiding rule is simple: **more compute should increase scale and fidelity, not be required to use Forka.**

See [ROADMAP.md](ROADMAP.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

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
python -m build
```
