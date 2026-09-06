# v0.2 proposal: scenario and model runtime

Written after reviewing main at `3b3dbc8`, before runtime implementation.
The v0.1 kernel, budgets, cache, router, packaging, examples and 32 tests were
reviewed; the baseline tests pass. This proposal adds a small composition layer
to the existing kernel. Forka orchestrates domain models; scientific libraries
remain responsible for their own numerical methods.

## Public boundaries

| Abstraction | Responsibility |
| --- | --- |
| `Model` | Structural protocol: `step(state, rng)` returns weighted transitions. No inheritance required. |
| `Agent(Model)` | Concrete, composable model with a name, environment and policy. Observes, chooses weighted actions, and applies each action to the shared environment state. |
| `Observation` | Data view supplied to a policy by an environment. |
| `Action` | Named intent, domain payload and nonnegative probability weight. |
| `Environment` | Protocol: `observe(state, actor)` and `apply(state, actor, action, rng) -> State`. No mutable world hidden inside the environment. |
| `Scenario` | Binds one model (possibly a domain-defined composite), initial state, config and optional evaluator; `run()` delegates to the existing kernel. |
| `Evaluator` | Callable protocol `evaluator(state) -> float`; scores initial and generated states before pruning. Without it, existing state scores are preserved. |
| `StateAdapter[T]` | Optional external boundary: `to_state(external)` and `from_state(state)`. External libraries are imported by adapters, never by the kernel. |

The agent policy is a callable `(Observation, random.Random) -> Iterable[Action]`.
An empty policy terminates the branch like an empty v0.1 step. Zero-weight actions
are ignored without applying them; weights are validated before environment work.
Environment application produces one state per action and may use the supplied
RNG. Enumerated environmental uncertainty can be implemented by a generic Model;
there is no requirement to encode scientific evolution as agent actions.

The multi-agent example will use an explicit alternating-turn composite Model,
with attacker and defender Agent instances sharing a stateless Environment.
Turn, actor memory and all evolving variables belong in branch State. There is no
implicit scheduler, global registry or requirement that every entity be an Agent.

## Execution, identity and observability

The existing `Simulation(step, config).run(initial)` path remains available.
`Scenario` adapts `model.step` into that path. Its optional evaluator changes only
scores, retaining the existing probability-first pruning and score tie-break.
Exceptions from domain code and evaluators propagate; they are not successful
simulation results. Evaluator outputs must be finite numbers.

State fingerprinting is an opt-in SHA-256 identity over a versioned, canonical,
type-tagged representation of data, score and terminal flag. Mapping keys are
sorted; list order and scalar types are significant. Supported data consists of
string-keyed dictionaries, lists, strings, integers, finite floats, booleans and
None. Unsupported objects, cycles and non-finite numbers are rejected explicitly,
never stringified or hashed using Python's randomized hash. Adapters must convert
arrays, dates, custom types and domain units explicitly. Fingerprints exclude
history and RNG state: they identify state snapshots, not replay checkpoints.
No deduplication is introduced because equal snapshots can have different paths.

Results add structured string-enum `termination_reasons` and a statistics value.
Reasons can include both pruning events (`max_branches`, `min_probability`) and
one final stop cause (`completed`, `max_steps`, `max_runtime`, `no_branches`).
Pruning reasons appear once in first-occurrence order; the final cause is last.
Legacy `truncated`, timing, branch histories and best-result selection remain.

Statistics definitions:

- `branches_explored`: positive-weight successor candidates generated, including
  candidates later pruned; excludes the root, zero weights and empty-step stops.
- `branches_pruned`: positive-weight candidates dropped by threshold/underflow,
  plus frontier entries removed by the branch cap, counted when removed.
- `peak_active_branches`: peak retained frontier size after pruning, including
  terminal entries and the initial root; does not claim to measure temporary memory.
- `retained_probability_mass`: sum of final branch probabilities; never renormalized
  after pruning, and may include unfinished branches after a budget stop.
- `runtime_seconds`: the same measured wall time as legacy `elapsed_seconds`.

## Reproducibility and future hosts

Models, policies and environments must treat supplied state/payloads as immutable
and put evolving state in returned State values, not mutable service objects.
The existing shallow-copy State helpers are unchanged. Determinism requires stable
transition order, a seed, and use of the supplied RNG; global randomness, external
side effects and wall-clock cutoffs can break replay. Runtime limits remain
cooperative between step calls. Large transition iterables are still materialized.

The library API has no CLI, HTTP, process or GPU dependency. Future CLI/server
adapters can construct a Scenario and serialize results; future workers can load
explicit model/config identifiers and portable state via adapters. Python callables
are not assumed to be serializable. Stable per-branch RNG streams, model versioning,
checkpoint formats and worker scheduling must be designed before distribution;
v0.2's serial traversal RNG does not promise scheduling-independent replay.

## Scope and compatibility

Implement protocols/value objects, Agent composition, Scenario, evaluator wiring,
fingerprinting, result reasons/statistics, and agent/non-agent examples with tests.
Use only the standard library at runtime. No REST, CLI, scenario schema,
distributed runtime, GUI, Docker, mandatory model provider or scientific package.

Existing v0.1 constructor arguments, state semantics, normalized weights, pruning
order, step counting and seeded results remain functional. New result fields are
appended with defaults; fingerprint validation applies only when explicitly used.
No intentional backward-incompatible behavior change is planned. Serialized
dataclass dictionaries gain fields, so consumers expecting an exact result field
set must account for additive metadata. Package version becomes `0.2.0.dev0` to
identify this unreleased foundation.
