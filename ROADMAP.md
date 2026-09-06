# Forka Roadmap

## v0.1 - Foundation
- [x] CPU-first branching kernel
- [x] probability normalization and branch pruning
- [x] runtime/branch budgets
- [x] seeded reproducibility
- [x] memory cache and compute-budget primitives
- [x] model-agnostic router interface
- [x] tests and CI

## v0.2 - Agent runtime
- [ ] Agent, policy, goal, observation, and action abstractions
- [ ] Event queue and environment API
- [ ] pluggable scoring/evaluation
- [ ] branch deduplication by state fingerprint
- [ ] CLI and JSON scenario format

## v0.3 - Intelligence + knowledge
- [ ] local-model adapter
- [ ] cloud-provider adapters as optional extras
- [ ] research-provider interface with source provenance
- [ ] persistent SQLite cache
- [ ] intelligent routing by complexity/cost/privacy

## v0.4 - Scale
- [ ] multiprocessing branch execution
- [ ] adaptive pruning
- [ ] hardware detection and automatic profiles
- [ ] distributed worker protocol prototype

## Long-term
Forka should let the same scenario run on a laptop, a gaming PC, or a cluster, with the engine automatically trading simulation breadth/depth against available compute and user-defined budgets.
