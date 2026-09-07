# v0.3 proposal: branch efficiency, experiments and measurement

Baseline: main `98c5f09` (merged v0.2), all 79 tests passing. Reviewed the complete
tracked repository, v0.1 kernel behavior and v0.2 contracts before implementation.
Model remains the primary interface. Agent, AI and external libraries are optional.
Implement using the standard library only, on `v0.3-efficiency-benchmarks`.

## Branch equivalence and pipeline

Add `SimulationConfig.deduplication`, default `"none"`. The only enabled mode is
`"state_markov"`: the caller explicitly asserts that equal complete State values
at equal branch depth have identical future transition distributions and scoring.
All changing model memory, time and policy state must be in State. The model and
evaluator must be deterministic functions of this state; shared RNG sampling,
hidden mutable model state, path-dependent behavior and external side effects do
not satisfy this contract. Forka cannot infer or automatically verify purity.
Enumerated probabilistic transitions are compatible; traversal-RNG draws generally
are not. Disable merging for such models. No custom coarse equivalence key in v0.3.

Merge within each newly constructed frontier using `(state.fingerprint(), steps)`.
The existing fingerprint includes data, score and terminal status; its canonical
format remains unchanged. Equal terminal snapshots at different depths stay
separate. Sum probabilities without renormalization. Preserve first-seen history
as an explicitly representative path; `Branch.path_count` sums represented paths
and propagates to children. Do not retain an exponentially growing list of histories.
Fingerprint validation is only required when merging is enabled.

With merging enabled, score positive candidates, merge, then threshold and cap.
This lets individually small paths survive as one sufficiently probable state.
With default options, retain v0.2's threshold-before-evaluator behavior and stable
probability/score ordering. Zero-weight/underflow paths are never executed further.
The kernel still materializes each model's transitions and one candidate frontier;
merging reduces subsequent expansions, not arbitrary generator resource use.

## Pruning boundary

`PruningStrategy.select(branches) -> Iterable[int]` returns indices of retained
entries. Supply `NoPruning()`, `MinProbability(threshold)` or `TopK(k)` to the new
optional `pruning` argument on Simulation/Scenario. Custom strategies select a
subset without modifying states or probabilities. Validate indices for range,
uniqueness and integer type; snapshot input is a tuple, but nested data is still
read-only by contract. The runtime preserves input ordering before its stable
probability/score sort, so strategy index ordering does not change RNG scheduling.

`pruning=None` retains both existing config-based threshold and cap. An explicit
strategy replaces these two limits (including NoPruning); step/runtime limits still
apply. Record this override in metadata. Never renormalize after selection. Record
removed mass and branch counts, and use structured pruning causes, including a new
`custom_pruning` cause for user strategies. These are trusted Python extensions.

## Statistics and provenance

Keep existing statistics fields and meanings. Add states visited (actual model
step invocations, not unique hashes), transitions generated (all yielded outputs,
including zero weights), branches retained (final frontier), branches merged
(redundant entries absorbed), states deduplicated (duplicate groups per frontier),
absorbed probability mass (mass of redundant entries, summed across merge events),
removed probability mass (mass pruned), and throughput. `branches_generated` aliases
the existing positive-candidate `branches_explored` count. Absorbed mass can exceed
one across depths; it is not lost probability. Retained plus removed mass should
equal one within floating-point tolerance. Underflow cannot recover subnormal mass.
Peak active means the retained frontier, not an RSS/memory measurement.

New observational fields do not participate in legacy SimulationStatistics equality
so existing five-field comparisons remain valid; compare dataclass dictionaries
when all metrics matter. Result metadata is also observational. Keep legacy
constructor positions, best(), truncation and default probability semantics.

## Experiments and local CPU workers

`Experiment(scenario, runs=1, seed=0, workers=1, keep_results=True).run()` creates
independent Scenario runs with seeds derived by a versioned SHA-256 scheme from
master seed and run index. It overrides the scenario seed deliberately. Each index
has the same seed regardless of scheduling or worker count. Accept zero runs as an
empty result with zero totals and undefined (None) means; reject negative counts,
booleans used as counts, and workers less than one.

Use ProcessPoolExecutor with explicit `spawn`, including on POSIX, for workers > 1.
Default is one worker. Parallel scenarios must be picklable with importable model,
policy/evaluator/adapter definitions; fail clearly before starting workers if not.
Require an `if __name__ == "__main__"` entry point. Respect Windows process limits.
Use at most min(workers, runs) processes and record requested/effective counts.
No threads, cluster runtime, automatic GPU selection or third-party pool library.

Each serial run deep-copies its scenario; each worker run loads a fresh serialized
scenario. Model instances therefore start fresh per run. Functions/closures and
module globals cannot be reset generically: they must be pure or manage explicit
state. Workers must not rely on PID, global RNG, wall clock or task order. Timings
and metadata naturally differ between workers; semantic branches/results must not.
External side effects are not rollback-safe and should use isolated test systems.

Results expose ordered seeds, optional individual results, aggregate counts, mean
probability retention/removal, wall time and throughput. keep_results=False drops
branch results after aggregation, but retains ordered seeds for replay. This is a
minimal execution abstraction, not confidence intervals or a parameter-sweep DSL.
Runtime budgets remain cooperative; failures propagate, not partial success.

## External systems and reproducibility

`SystemUnderTest[Input, Output].run(value)` and `CallableSystem(function)` wrap
arbitrary Python objects/functions. A `SystemModel` composes the system with input
and output adapter callables to map State into external data and responses back
into State. One call per expanded branch, no implicit retries or network adapter.
The system need not know about Agent or Forka. Existing StateAdapter remains useful.

Produce a structured record with Forka/Python/platform version, optional package
source Git revision/dirty status, seed, configuration, requested/effective workers,
model/scenario/evaluator identifiers and deterministic settings. Infer identifiers
from qualified Python names only. Never serialize arbitrary object repr, model
attributes, State, environment variables, hostnames, usernames or secrets.
Custom pruning parameters/model configuration require separate caller documentation;
do not pretend that qualified names are sufficient executable checkpoints.

Git detection applies only to this package's source checkout, not a caller's repo;
installed wheels report unavailable. Cache this source snapshot once per process
to avoid subprocess overhead on every run, and document that later source edits
require a fresh process. Wall time excludes metadata construction in Simulation;
Experiment wall time includes run dispatch, copying and worker startup/teardown.

## Measurement plan and limitations

Add a CPU benchmark runner with repeated, warmed measurements and raw JSON output.
Compare the identical convergent branching workload without/with dedup, cap/threshold
pruning, and a divergent control where hashing cannot help. Include workers 1/2
experiments on a fixed workload to reveal startup overhead. Report all repetitions,
median runtime, generated/retained/merged/pruned branches, probability retention,
throughput, and a separately measured tracemalloc peak (Python allocations, not RSS).
Record CPU model/count, OS, Python, source revision/dirty flag and exact commands.
Benchmarks must assert probability/result invariants, never assert speedup or hide
slow cases. No services, datasets, credentials or paid compute.

Test default compatibility, equivalence/depth constraints, merging/provenance/mass,
pruning extension validation, malformed config, stable seeds, spawn worker parity,
empty experiments, external adapters, metadata hygiene and benchmark invariants.
Run all old/new examples, tests, Ruff, source/wheel builds and Windows/Linux CI.

v0.4 direction: measured streaming frontier construction and cancellation first;
then branch-local RNG/checkpoint design and richer adapter reproducibility. Do not
add REST, distribution, LLM SDKs, provider gateways or infrastructure in v0.3.
