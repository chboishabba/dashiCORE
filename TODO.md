# TODO — Sprints 0 to 2

- [x] Sprint 0: implement core modules (`carrier`, `kernel`, `defect`, `admissibility`, `mdl`, `hierarchy`) with validation + shape invariants.
- [x] Sprint 0: add deterministic mock kernels under `dashi_core/testing/mock_kernels.py`.
- [x] Sprint 0: mirror spec test layout under `tests/` with invariants + violations.
- [x] Sprint 1: introduce backend base/registry and refactor CPU ops to use it; add parity/determinism tests.
- [x] Sprint 1: add backend violation tests for forbidden behaviors.
- [x] Sprint 2: add accelerated backend implementation matching CPU semantics; declare capabilities + precision policy.
- [x] Sprint 2: add numeric drift + cross-backend reproducibility tests.
- [x] Run full pytest suite and document results.
- [ ] Integration prep: document dashiBRAIN support/neutral policy and mapping (done in docs/dashibrain_core_integration.md).
- [x] Add legacy adapter utilities for signed ↔ Carrier (with neutral mask) and tests.
- [ ] Define GraphContext + GraphHopKernel wrapping existing graph logic (CPU parity).
- [ ] Add flow parity + violation guard tests for graph kernels.
- [x] Keep GPU adapters outside CORE; plan Vulkan adapter using gpu_common_methods.py (post-Sprint2) — Vulkan adapter scaffold + dispatcher hooks landed.
- [x] Add dependency manifests (`requirements.txt`, `requirements-dev.txt`) and document install command.
- [x] Add PQ vs dense benchmark harness (JSONL output, size/sparsity/backend/block sweeps) and PQ block-size heuristic.
- [ ] Add GPU/CPU kernel partitioning tests (disjoint/halo/scheduling) to lock parallel recomposition semantics.
