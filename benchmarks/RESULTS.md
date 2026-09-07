# Measured CPU results: v0.3 development

Measured on 2026-09-07 with an Intel Core Ultra 9 185H (16 cores, 22 logical CPUs),
Windows 11 AMD64 and Python 3.12.14. CPU name/core counts were read from Windows
hardware inventory; the raw report also records Python's processor descriptor.
Forka version: 0.3.0.dev0. Source commit:
`1c07c3ee372791e57af4c4ee8ee33eee10f09393`, clean working tree at measurement time.
Later report/document commits do not change the measured implementation.

```bash
python -m pip install -e ".[dev]"
python -m benchmarks.run --depth 7 --repeats 5 --runs 128 --work 10000 --output benchmarks/results/windows-cpu.json
```

[Raw final results](results/windows-cpu.json) include all five repetitions per case,
configuration and provenance. One warmup per case; times below are medians. Memory
is peak tracemalloc Python allocation from a separate run, not RSS. See the
[methodology](README.md). Assertions verify probability conservation, full-mode
state distributions/represented path counts and serial/parallel output parity.

## Branching workload

Each tick has four equally weighted alternatives, up to seven ticks. The convergent
case preserves only tick and position; the divergent control also preserves path
identity, preventing merging. All cases below ran, including unfavorable ones.

| Workload / pruning | Dedup | Median ms | Generated | Retained entries | Merged | Pruned | Retained mass | Peak Python MiB | Transitions/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Convergent / none | off | 54.1 | 21,844 | 16,384 | 0 | 0 | 1 | 11.6 | 404,000 |
| Convergent / none | on | 1.62 | 196 | 15 | 133 | 0 | 1 | 0.0309 | 121,000 |
| Convergent / top 64 | off | 2.53 | 1,108 | 64 | 0 | 768 | 0.00390625 | 0.150 | 438,000 |
| Convergent / top 64 | on | 1.50 | 196 | 15 | 133 | 0 | 1 | 0.0308 | 130,000 |
| Convergent / threshold 0.0001 | off | 50.6 | 21,844 | 0 | 0 | 16,384 | 0 | 10.5 | 432,000 |
| Convergent / threshold 0.0001 | on | 1.57 | 196 | 13 | 133 | 2 | 0.999878 | 0.0309 | 125,000 |
| Divergent / none | off | 68.4 | 21,844 | 16,384 | 0 | 0 | 1 | 13.5 | 319,000 |
| Divergent / none | on | 472 | 21,844 | 16,384 | 0 | 0 | 1 | 17.1 | 46,300 |

Full convergent merging is about **33.4 times faster** here, preserving all mass and
all 16,384 represented paths with 15 frontier entries. Model invocations fall from
5,461 to 49. Both modes represent the same 15 distinct final states; baseline
entries include duplicate states. Hashing makes each generated transition more
expensive (lower transitions/s), but avoiding expansions wins on total runtime.

The divergent control is about **6.91 times slower** with merging and uses more
Python allocation. Equal-state convergence is essential; do not turn merging on
universally. The top-K speed improvement without merging loses 99.6% of probability
mass. A per-path threshold of 0.0001 loses every path at depth seven, whereas
merged-state pruning loses only 1/8192 mass. These are explicit semantic differences,
not interchangeable accuracy settings or a renormalized presentation of results.

## Repeated CPU experiments

128 independent seeded runs; each evaluates 10,000 sin/cos terms and one RNG
observation. Master seed 42, five samples, pool startup/shutdown included.

| Workers | Median seconds | Simulations/s | Retained mass per run |
| ---: | ---: | ---: | ---: |
| 1 | 0.252 | 507 | 1 |
| 2 | 0.343 | 373 | 1 |

Two workers are about **1.36 times slower** for this workload. Startup, per-process
metadata collection, serialization and coordination outweigh the parallel work.
Workers=1 remains the default. Larger workloads may benefit, but this run does not
establish that crossover or justify any claim of universal parallel speedup.

## Reproducibility and limits

- This is one development laptop/session, not a hardware survey. OS scheduling,
  background activity, thermals and power settings were not controlled. Raw
  repetitions expose variation; reported precision is deliberately limited.
- Counts and represented distributions, not timing, are correctness assertions.
  Frontier/state memory is still materialized; this does not prove bounded memory.
- Merging assumes exact state/depth-only future behavior. The workload enumerates
  probabilities and has no hidden state or traversal-RNG draws.
- The experiment uses deterministic standard-library arithmetic on this Python
  platform. External systems/scientific libraries require their own reproducibility
  controls. Timing and metadata are not expected to match across workers or hosts.
- Kernel statistics use the legacy monotonic clock, which can round very short
  runs to zero on this environment. Benchmark timing uses the higher-resolution
  outer perf_counter; the raw per-run statistics are from the final repetition,
  while headline rates use median measured duration.

An [initial raw run](results/windows-cpu-initial.json) is retained for transparency.
It used `d48e588c03832717472b4fe75c27eeb273296bff`. That version incorrectly labeled
an experiment's template as unseeded in metadata, although actual run seeds were
already derived correctly. The final run above followed the metadata correction;
the simulation/pruning/parallel algorithms did not change between these runs.

Recommended next milestone: streaming frontier construction and bounded experiment
submission, with cancellation and measurements across workload sizes. Design
branch-local RNG/checkpoint semantics before broadening deduplication contracts.
