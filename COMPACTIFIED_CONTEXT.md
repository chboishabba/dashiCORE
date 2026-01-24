# dashiCORE — Compactified Context

## Scope
- Progress Sprints 0 → 2 (stop before Sprint 3).
- Implement minimal core types + invariants, backend abstraction with CPU + accelerated-compatible backend, and parity/numeric drift tests.

## Intent (Pre-implementation)
- Core modules: `dashi_core/{carrier,kernel,defect,admissibility,mdl,hierarchy}.py` with pure, shape-preserving operations and explicit validation.
- Testing helpers: `dashi_core/testing/mock_kernels.py` with deterministic mocks only for tests.
- Backend layer: `dashi_core/backend/{base,cpu,registry}.py` plus an explicit accelerated backend candidate matching CPU semantics.
- Tests mirror spec: `tests/{carrier,kernel,defect,admissibility,mdl,hierarchy,backend,reproducibility,violations}/...`.
- Precision + determinism locked to CPU reference; backends must declare capabilities and fail loudly on unsupported features.
- Integration policy for dashiBRAIN: support = node/channel existence (constant True); neutrality handled via external masks, not support flips.

## Assumptions
- Python 3.11+, NumPy + PyTest available.
- No GPU available; accelerated backend = CPU-parity implementation using the same protocol.
- No changelog file yet; will note omission.

## Open Questions
- Whether to add packaging metadata now or defer until API stabilizes. Current plan: minimal package layout without publishing artifacts.

## Current Status
- Core modules, backend layer (CPU + accelerated), and invariant tests implemented.
- Tests: `python -m pytest` (31 passed).
- Integration doc drafted at `docs/dashibrain_core_integration.md`; TODO updated for adapter and graph kernel parity work.

## Latest GPU/vkFFT runs (field reports)
- N=640, steps=30 (`run_v4_snapshots.py`, backend=vulkan, fft-backend=vkfft-vulkan, dtype=float64, RADV RX 580): completed; encode=16.79s, learn=0.029s, rollout≈0, decode_total=0.369s (0.061s/snap). Observed ~600 MB **system RAM** (RSS); GPU VRAM not measured.
- N=1024, steps=300 (same flags): aborted in encode due to NaNs (`RuntimeError: Non-finite values in omega before encoding`). Mitigate by reducing `dt`/`Cs` or lowering N.
- Kernel-only mode still requires a real `--z0-npz` path (keys: `z`, `mask_low`, `anchor_idx`); placeholders will raise `FileNotFoundError`.
