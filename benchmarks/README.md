# Forka CPU benchmarks

Run from the repository root after `python -m pip install -e ".[dev]"`:

```bash
python -m benchmarks.run --output benchmarks/results/local.json
python -m benchmarks.run --smoke
```

No credentials, data downloads, GPU or services are used. The default is depth 7,
five measured repetitions, and 128 experiment runs with 10,000 math operations per
run. Parameters are explicit flags: `--depth`, `--repeats`, `--runs`, `--work`.
The smoke run reduces them to 3, 1, 4 and 20. There are no speed thresholds in CI.

Every case gets one warmup, then all raw timed repetitions are saved. Simulation
timings use an outer perf_counter, so even short kernels get useful measurements.
One separate run measures peak Python allocation with tracemalloc; this is neither
RSS nor whole-process memory, and its overhead is excluded from reported speed.
Experiment samples include process pool startup/shutdown every time. The runner
asserts probability conservation and full-mode distribution/path-count equality,
plus identical seeded experiment outputs across worker counts.

Cases are deliberately paired:

- A convergent four-way lattice with two zero-displacement alternatives. Compare
  disabled/enabled merging with no pruning, TopK(64), and MinProbability(0.0001).
- A divergent control retaining path identity in State, so no branches can merge.
  This exposes hashing cost when there is no state convergence.
- A fixed numerical CPU workload at workers 1 and 2, to expose spawn/IPC overhead.

These synthetic cases illustrate a favorable and an unfavorable shape. They are
not evidence of universal speedup, calibrated scientific accuracy or a comparison
against other frameworks. Larger workloads may reverse worker-speed results;
small ones commonly favor workers=1. A memory-heavy branching model can still
exhaust memory because transitions/frontiers are materialized. Do not increase
depth casually: the full divergent tree has 4**depth leaves.

Raw reports include parameters, exact effective command, CPU descriptor/count,
Python/OS, source version/dirty state, timings, counters, retention and allocation
peaks. Hardware processor strings may be generic; measured reports can additionally
identify the CPU from local hardware inventory. No hostname or credentials appear.

See [RESULTS.md](RESULTS.md) for the measured development-machine results. Rerun on
your target machine; runtime and spawn costs are platform/workload dependent.
