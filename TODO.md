# TODO — Sprints 0 to 2

- [x] Sprint 0: implement core modules (`carrier`, `kernel`, `defect`, `admissibility`, `mdl`, `hierarchy`) with validation + shape invariants.
- [x] Sprint 0: add deterministic mock kernels under `dashi_core/testing/mock_kernels.py`.
- [x] Sprint 0: mirror spec test layout under `tests/` with invariants + violations.
- [x] Sprint 1: introduce backend base/registry and refactor CPU ops to use it; add parity/determinism tests.
- [x] Sprint 1: add backend violation tests for forbidden behaviors.
- [x] Sprint 2: add accelerated backend implementation matching CPU semantics; declare capabilities + precision policy.
- [x] Sprint 2: add numeric drift + cross-backend reproducibility tests.
- [x] Run full pytest suite and document results.
