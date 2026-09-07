# v0.3 efficiency and experiment API

Model first. Agent optional. AI optional. v0.3 builds on the existing runtime;
no GPU, network, API key or additional runtime dependency is needed.

## Optional branch deduplication

```python
from forka import Scenario, SimulationConfig, State, Transition


class CoinCount:
    def step(self, state, rng):
        tick = state.data.get("tick", 0) + 1
        # Two different decisions have the same outcome and future evolution.
        next_state = State({"tick": tick}, terminal=tick == 3)
        return [Transition(0.4, next_state, "a"), Transition(0.6, next_state, "b")]


result = Scenario(CoinCount(), config=SimulationConfig(
    seed=42, deduplication="state_markov"
)).run()
assert len(result.branches) == 1
assert result.best().path_count == 8
assert result.statistics.retained_probability_mass == 1
```

`deduplication="none"` is the default. `"state_markov"` is an explicit assertion
that exact state plus depth determines future behavior, with no traversal RNG
sampling, hidden evolving service state, path dependence or external side effects.
Enumerated uncertainty is fine. Do not enable this mode just because a stochastic
model has a seed. Equivalence uses the unchanged v0.2 fingerprint, including score,
terminal flag and exact scalar types, plus Branch.steps. Opaque external data must
be converted through adapters before hashing. No approximate equality is inferred.

The retained history is the first-seen representative, not the only path taken.
`path_count` counts represented paths; full histories are deliberately not retained.
Merging precedes threshold/cap pruning in this mode. Therefore threshold and top-K
operate on merged states, and can retain more probability than path-wise pruning.

## Pruning strategies

Pass `pruning=TopK(64)`, `MinProbability(0.001)` or `NoPruning()` to Scenario or
Simulation. An explicit strategy **replaces** config.min_probability and
config.max_branches; it does not replace step or runtime limits. `pruning=None`
preserves both v0.2 config limits. Zero-weight and underflow paths are still omitted.
`TopK(0)` is valid and returns an empty frontier; negative/non-integer K is invalid.

A custom PruningStrategy implements `select(branches) -> Iterable[int]`. It receives
a tuple and returns unique in-range indices, never modified probabilities/states.
The kernel validates indices and retains original relative order before its stable
probability/score sort. Do not mutate the nested Branch/State objects: extensions
are trusted Python code, not isolated plugins. Custom selection losses have the
structured `custom_pruning` cause. Removed mass is always recorded without
renormalization, including losses from built-in default limits.

## Experiment and CPU parallelism

```python
from forka import Experiment, Scenario, State


class Sample:
    def step(self, state, rng):
        return [(1, State({"value": rng.random()}, terminal=True), "sample")]


experiment = Experiment(Scenario(Sample()), runs=10, seed=42)
result = experiment.run()
assert len(result.results) == len(result.seeds) == 10
print(result.statistics.mean_retained_probability_mass)
```

For CPU processes use `workers=2` from a script guarded by
`if __name__ == "__main__":`. See `examples/experiment_runs.py`. Use importable
top-level models/functions rather than lambdas, nested classes or notebook-local
definitions. Spawn is used on Windows and POSIX; Windows inherits Python's
ProcessPoolExecutor worker limit (61). Forka requests min(workers, runs) workers,
with one worker as the sensible default. No process pool is created for zero or
one effective worker. The per-run scenario must also be picklable in parallel mode.

`run_seed(master_seed, index)` is a versioned SHA-256 derivation to a 64-bit integer.
Experiment overrides Scenario.config.seed per run. Same master seed/index means
the same run regardless of worker count or scheduling; adding runs preserves the
existing prefix. Returned results and seeds are in run-index order. Replay a run
by replacing Scenario.config.seed with its entry in result.seeds.

Each serial run deep-copies its Scenario; parallel workers load a fresh serialized
template for every run. Mutable model instance fields are reset this way, but
module globals, function closures, external resources and class-level state are
not generic resettable simulations. Keep them pure or provide an explicit wrapper.
Only the caller's own trusted Python objects are serialized; never accept untrusted
pickle bytes. Exceptions propagate; a failed experiment has no success result.

`keep_results=False` aggregates and releases individual branch results. Seeds are
still retained, and the process pool currently queues run tasks eagerly; this is
not a constant-memory streaming scheduler. All results are kept by default.
`runs=0` returns empty tuples, zero counts/rates and None probability means.
Experiments are the minimal foundation for future sweeps, not a statistics DSL.

## External systems under test

```python
import statistics
from forka import Scenario, State, SystemModel

model = SystemModel(
    statistics.mean,
    input_adapter=lambda state: state.data["samples"],
    output_adapter=lambda state, output: state.with_data(mean=output).stop(),
)
result = Scenario(model, State({"samples": [1, 2, 3]})).run()
assert result.best().state.data["mean"] == 2
```

SystemModel accepts a callable or an object with `run(value)` conforming to the
SystemUnderTest protocol. CallableSystem(function) explicitly adapts a function
to that object protocol. A callable object's __call__ takes precedence over run.
Adapters map arbitrary external inputs/outputs to State without requiring agents.
The output adapter must return a State and decides whether execution is terminal.
One system call occurs per expanded branch. There are no retries, network calls,
timeouts, authentication or provider SDKs hidden in this wrapper. Use isolated
test instances for effectful systems; forka does not undo calls, rate-limit them,
or guarantee reproducibility of external responses. Never merge effectful runs.

## Metrics and reproducibility

SimulationStatistics retains its five legacy fields. Additional counters:

| Field | Meaning |
| --- | --- |
| states_visited | Model step invocations; not a unique-state count |
| transitions_generated | All yielded transition entries, including zero weights |
| branches_generated | Alias of branches_explored: positive candidates |
| branches_retained | Final frontier size, including unfinished/terminal states |
| branches_merged | Redundant frontier entries absorbed |
| states_deduplicated | Groups containing duplicate entries, counted per frontier |
| absorbed_probability_mass | Mass of redundant entries across merge events; may exceed 1 over depths |
| removed_probability_mass | Pruned mass; retained + removed is 1 within floating-point tolerance |
| transitions_per_second | Transitions / kernel wall time; 0 when clock resolution gives zero time |

Peak active still means the post-pruning frontier, not memory usage. Observational
v0.3 fields do not participate in legacy SimulationStatistics equality. Use
dataclasses.asdict for complete metric comparisons. Timing is not deterministic.

ExperimentStatistics sums visited/generated/retained/merged/pruned counts and
deduplicated groups across runs, reports peak retained frontier, means of retained
and removed mass, effective workers, wall time, simulations/s and transitions/s.
Experiment wall time includes copying, dispatch, startup and shutdown. Kernel
wall time excludes reproducibility collection. Both include model/evaluator work.

`result.reproducibility` is a ReproducibilityRecord with version, optional source Git
commit/dirty flag, seed, config, qualified model/scenario/evaluator identifiers,
Python/platform and deterministic settings. Experiment records master and run
configuration; each retained simulation records its derived seed. No input state,
model attributes, object repr, environment variables, hostname, username or token
is collected. Built-in pruning parameters are recorded; custom pruning parameters
are explicitly marked unrecorded. This is metadata, not a replay checkpoint.

Git info comes only from the package source checkout and is cached once per
process; restart after source changes for a fresh snapshot. Wheels report None
rather than the caller's repository. Supply scenario.identifier for a meaningful
public scenario name; model/solver versions and non-public inputs remain the
researcher's responsibility. Seeds alone do not control global RNG, external
services, cross-platform numerical libraries or runtime cutoffs.

## Compatibility and v0.4

Default v0.2 execution remains functional, with all 79 baseline tests preserved.
New config/result fields have defaults, existing StateAdapter and Agent composition
are unchanged, and no runtime dependency was added. One validation tightening:
SimulationConfig.seed now enforces its declared int-or-None type, rejecting booleans,
strings and floats formerly accepted incidentally by random.Random. For portability,
explicitly convert old non-integer seeds yourself. Metadata and metric fields are
additive for consumers serializing dataclasses. The version is `0.3.0.dev0`.

Next: measured streaming frontier construction, bounded experiment submission and
cooperative cancellation; then branch-local RNG streams and restartable model
checkpoints. Exact fingerprints are not scientific tolerance comparisons. Do not
add distributed infrastructure or selective-inference providers before measuring
these local bottlenecks.
