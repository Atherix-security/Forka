# v0.2 API notes

All public names below are importable from `forka`. Python 3.10+ and the standard
library are sufficient at runtime; pytest, Ruff and build are development extras.

## Composition

```text
Scenario(model, initial_state=State(), config=SimulationConfig(), evaluator=None)
Agent(name, environment, policy)
Action(name, data={}, probability=1.0)
Observation(data={})
Simulation(step, config=None, *, evaluator=None)
```

The defaults above describe constructor behavior; mutable defaults are actually
created separately for each instance by dataclass factories.

`Model.step(state, rng)` returns an iterable of Transition values or legacy
`(weight, state, label)` tuples. `Agent` is a concrete Model implementation, not a
required parent class for domain entities. A policy takes `(observation, rng)` and
returns Action values. Environment has two methods:

- `observe(state, actor: str) -> Observation`
- `apply(state, actor: str, action, rng) -> State`

The environment receives the same parent state for each action alternative.
Implementations must return new states rather than mutate inputs. A policy with
no actions terminates its branch; invalid weights raise before any action is
applied, and zero-weight actions are skipped. Each transition history label is
`actor:action_name`; these are display labels, not identifiers to parse or route.

Scenario does not impose a scheduler. Compose models in domain code and store
turn/timing variables inside State. Reusing a Scenario reinitializes the kernel RNG
from its config seed. Reusing mutable model/service state is the caller's
responsibility: neither copying nor resetting third-party models is implicit.

## Evaluation and results

`Evaluator` is a callable protocol `(state) -> float`; plain functions and callable
objects work. A provided evaluator sets the initial score and scores candidate
states that pass probability filtering, before the branch cap is applied. It also
scores the terminal state produced by an empty step. It must return finite real
numbers (not booleans). Lower-level state scores remain untouched when no evaluator
is supplied. Evaluation does not change probability-first pruning; score breaks
ties, while `result.best()` selects by score by default.

`result.termination_reasons` contains enum members from `TerminationReason`:
`MAX_BRANCHES`, `MIN_PROBABILITY` for pruning events, followed by exactly one final
cause: `COMPLETED`, `MAX_STEPS`, `MAX_RUNTIME` or `NO_BRANCHES`. A completed frontier
can still be truncated because it lost probability mass. An empty step means a
terminal branch, while `NO_BRANCHES` means no retained frontier remains. Causes
have stable string values suitable for transport; domain exceptions propagate.

`result.statistics` is a frozen `SimulationStatistics` with `branches_explored`,
`branches_pruned`, `peak_active_branches`, `retained_probability_mass`, and
`runtime_seconds`. Exact counting definitions are in the architecture proposal.
An unexpanded root has explored/pruned counts of zero, a peak of one and retained
mass one. Carried terminal states and empty-step stops are not counted again as
generated branches. Pruned paths are counted at their removal point, not as all
hypothetical descendants. Runtime and retained mass describe the returned result,
including unfinished paths when a budget stops execution.

## External adapter boundary

`StateAdapter[T]` describes `to_state(value: T) -> State` and
`from_state(state: State) -> T`. Domain packages own their imports, units, dtype
conversion, schemas and validation. They can wrap library calls behind Model.step
and use an adapter at entry/exit. Forka does not call adapters automatically or
inspect model internals. For example, a tuple-based external vector can use:

```python
from forka import State, StateAdapter


class VectorAdapter:
    def to_state(self, value):
        return State({"coordinates": list(value)})

    def from_state(self, state):
        return tuple(state.data["coordinates"])


adapter: StateAdapter[tuple[float, ...]] = VectorAdapter()
initial = adapter.to_state((1.0, 2.0))
assert adapter.from_state(initial) == (1.0, 2.0)
```

For a NumPy or other scientific integration, import the library in your adapter
package and explicitly encode values (and any necessary dtype/unit metadata).
No particular scientific dependency or serialization format is mandated here.

## State identity

`state.fingerprint()` and `state_fingerprint(state)` return the same SHA-256 hex
digest. Canonical format `forka-state-v1` includes data, score and terminal flag;
dict insertion order is irrelevant. Supported values are exact built-in dicts
with string keys, lists, str, int, finite float, bool and None. Types are
significant: `1`, `1.0` and `True` have different identities, as do signed float
zeros. Floats use their exact hexadecimal representation. Strings use their
original code points, without Unicode normalization. Cycles and non-finite values
raise ValueError; unsupported values and non-string keys raise TypeError.

Fingerprints are recomputed, not cached: v0.1 State.data remains mutable.
This identity is not a numerical tolerance comparison or a replay checkpoint and
does not include model version, branch history or RNG state. It is not reused as
the existing permissive MemoryCache.key, whose behavior remains unchanged.

## Compatibility and technical debt

No intentional breaking change to the v0.1 public execution API. Existing
constructor positions, legacy transition tuples, probability normalization,
pruning order, shallow state helpers and step accounting remain. Results gain
defaulted fields; callers serializing dataclasses must accept the added metadata.
The development version is `0.2.0.dev0`, not a published release.

Known limits intentionally retained for this foundation:

- State, Action and Observation contain mutable nested data. Branch isolation
  requires functional domain code; deep copying every scientific state would be
  expensive and is not automatically imposed.
- Runtime budgets are cooperative. A slow model/evaluator or unbounded transition
  iterable can exceed time/memory limits; peak_active_branches is not memory usage.
- One serial RNG is shared across a run. Stable transition order is required;
  scheduling-independent streams and checkpoint serialization are future work.
- Fingerprints do not deduplicate branches or establish scientific equivalence.
- Environment.apply returns one state per action. Use a generic Model for explicit
  multi-outcome environment evolution, or the supplied RNG for sampling.
- Model registration/versioning, restartable scientific solvers, richer agent
  memory/goals, cancellation and adapter registries are deferred.
- Existing compute budgets, cache and intelligence router remain opt-in primitives.
  No automatic inference, external I/O, REST, distributed scheduler or scenario
  JSON/YAML schema is introduced.
