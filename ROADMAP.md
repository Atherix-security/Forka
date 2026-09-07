# Forka Roadmap

## v0.1 - Foundation
- [x] CPU-first branching kernel
- [x] probability normalization and branch pruning
- [x] runtime/branch budgets
- [x] seeded reproducibility
- [x] memory cache and compute-budget primitives
- [x] model-agnostic router interface
- [x] tests and CI

## v0.2 - Scenario/model foundation (in development)
- [x] generic Model and Scenario, independent of agents
- [x] optional Agent(Model), policy callable, Action and Observation
- [x] shared Environment interface with branch-local world state
- [x] pluggable scoring/evaluation
- [x] deterministic state fingerprinting
- [x] structured termination reasons and result statistics
- [x] external state/model adapter boundary
- [x] agent and non-agent scientific-style examples

## Follow-on runtime work
- [ ] richer agent goals and memory
- [ ] event scheduling
- [ ] branch deduplication by state fingerprint
- [ ] CLI and scenario format design

## v0.3 - Efficiency and experiments (in development)
- [x] explicit state-equivalence deduplication with probability/provenance accounting
- [x] pruning strategy boundary and built-in threshold/top-K/no-pruning
- [x] reproducible Experiment runs and local spawn CPU workers
- [x] external callable/object systems-under-test boundary
- [x] efficiency metrics, reproducibility records and CPU benchmark suite

## v0.4 - Measured local efficiency and reliability
- [ ] streaming frontier construction and bounded task submission
- [ ] cooperative cancellation and per-run failure records
- [ ] branch-local RNG and checkpoint semantics
- [ ] domain adapter/model versioning and reproducibility extensions

## Later - Optional intelligence + knowledge
- [ ] local-model adapter
- [ ] cloud-provider adapters as optional extras
- [ ] research-provider interface with source provenance
- [ ] persistent SQLite cache
- [ ] intelligent routing by complexity/cost/privacy

## Later - Scale
- [ ] evaluate branch-level parallelism after measuring experiment-level workers
- [ ] adaptive pruning
- [ ] hardware detection and automatic profiles
- [ ] distributed worker protocol prototype

## Long-term
Forka should let the same scenario run on a laptop, a gaming PC, or a cluster, with the engine automatically trading simulation breadth/depth against available compute and user-defined budgets.
