# dashiCORE

> **Canonical mathematical primitives for DASHI-based systems**
> *Stable core objects. No domain assumptions. No experiments.*

---

## 0. Purpose & Non-Goals

### Purpose

`dashiCORE` defines the **minimal mathematical substrate** shared across all DASHI-derived projects.

It exists to:

* Prevent re-definition drift of core objects
* Provide a **single source of truth** for admissibility, carriers, kernels, and defects
* Allow downstream projects to focus on **domain-specific structure**, not foundations

If a concept appears in **two or more projects**, it probably belongs here.

### Explicit Non-Goals

`dashiCORE` does **not**:

* Implement domain logic (physics, biology, law, ML, graphics, etc.)
* Contain experimental algorithms or heuristics
* Optimize for performance over clarity
* Encode UI, visualization, or IO concerns

> Think **axioms + datatypes + invariants**, not pipelines.

---

## 1. Conceptual Scope

`dashiCORE` formalizes the following **foundational objects**:

* **Carriers**

  * Balanced ternary carrier ( T = {-1, 0, +1} )
  * Support vs sign factorization
* **Kernels**

  * Local consistency operators
  * Closure / projection operators
* **Defects**

  * Inconsistency measures
  * Contractive defect dynamics
* **Admissibility**

  * Redundancy quotients
  * Gauge / diffeomorphism invariance as equivalence
* **MDL Principle**

  * Complexity as a selection pressure, not a loss
* **Hierarchy**

  * M-levels (M3 / M6 / M9 …)
  * Tensor / bitensor composition rules

These definitions are **domain-agnostic** and must remain so.

---

## 2. Core Mathematical Objects

### 2.1 Carrier

Defines the canonical signed carrier:

* Balanced ternary field
* Explicit separation of:

  * **Support** (existence)
  * **Orientation / sign**

This factorization is **non-optional** and enforced at the type level where possible.

---

### 2.2 Kernel

A **kernel** is a local operator that:

* Acts on admissible neighborhoods
* Projects states toward local consistency
* Is idempotent or contractive under iteration

Kernels define **what counts as structure** in a system.

---

### 2.3 Defect

A **defect** is a measurable violation of kernel consistency.

Formally:

* Zero defect ⇒ fixed point
* Positive defect ⇒ inconsistency mass
* Defects are:

  * Local
  * Composable
  * Reducible under kernel action

Defect geometry is first-class.

---

### 2.4 Admissibility

Admissibility is treated as a **quotient**, not a constraint.

* Coordinates are gauge
* Only invariants survive
* Two states are equivalent if related by an admissibility transformation

This replaces ad-hoc “invariance handling” downstream.

---

### 2.5 MDL Selection

Minimum Description Length (MDL) is encoded as:

* A preference over representations
* A pressure toward lower-order jets / simpler kernels
* A tie-breaker between admissible representations

MDL is **structural**, not statistical.

---

### 2.6 Observations (Fingerprint)

`StateFingerprint` is a backend-invariant observational helper:

* Type: `Carrier[Ω] -> UInt64`
* Deterministic, pure, shape- and admissibility-invariant
* Purpose: witness equality across backends without transferring full state
* Non-goal: not semantic; never used in kernel/defect/MDL logic
* Implementations: CPU NumPy reduction; Vulkan GPU-side hash reduction (device-local) used in benchmarks to avoid full readback

Treat observations as **parity witnesses only**, not algorithmic signals.

---

## 3. Hierarchy & Composition

### M-Levels

`dashiCORE` defines the **rules**, not the content, of hierarchy:

* M3: local primitives
* M6: bitensors of M3
* M9: tensors of M6
* …

Each level obeys:

* Composition laws
* Projection rules
* Defect propagation constraints

No project may redefine M-level semantics independently.

---

## 4. What Lives Here vs Elsewhere

### Lives in dashiCORE

* Mathematical definitions
* Canonical dataclasses / types
* Invariant checks
* Minimal reference implementations
* Proof-oriented helpers

### Lives Outside

* Domain kernels (CFD, ROM, connectomics, law, etc.)
* Visualization
* Performance optimizations
* Dataset-specific code
* Training loops / solvers

---

## 5. Stability & Versioning

This repo prioritizes **semantic stability**.

* Breaking changes are rare and deliberate
* Definitions change only with:

  * Formal justification
  * Cross-project impact review
* Experimental ideas belong elsewhere until proven invariant-worthy

Expect slow evolution.

---

## 6. Relationship to Other Repos

`dashiCORE` is:

* Imported by higher-level projects
* Never imports them in return
* The bottom of the dependency graph

Downstream projects **must not fork or shadow core definitions**.

---

## 7. Minimal Usage Example (Conceptual)

```python
from dashi_core.carrier import TernaryField
from dashi_core.kernel import Kernel
from dashi_core.defect import defect_energy

state = TernaryField(...)
kernel = Kernel(...)

projected = kernel(state)
Δ = defect_energy(state, projected)
```

No domain meaning is implied here — only structure.

---

## 8. Design Philosophy (Short)

* **Structure before semantics**
* **Invariance before coordinates**
* **Compression before optimization**
* **Defects are information**
* **Hierarchy is compositional, not emergent**

If something feels clever, it probably doesn’t belong here.

---

## 9. Roadmap (Initial)

* [ ] Formal carrier + support/sign types
* [ ] Kernel interface + contractivity checks
* [ ] Defect geometry primitives
* [ ] Admissibility group abstraction
* [ ] MDL accounting utilities
* [ ] M-level composition helpers
* [ ] Minimal reference tests (invariance + idempotence)

---

## Implementation Scaffold (Sprints 0–2)

* Code lives under `dashi_core/` (`carrier.py`, `kernel.py`, `defect.py`, `admissibility.py`, `mdl.py`, `hierarchy.py`); mock kernels reside in `dashi_core/testing/mock_kernels.py`.
* Backends are explicit: `dashi_core.backend` exposes CPU + accelerated-compatible backends with `set_backend("cpu"|"accelerated")`; selection is deterministic and global.
* GPU adapters/backends stay outside CORE: `gpu_vulkan_adapter.py` (adapter + kernel wrapper), `gpu_vulkan_backend.py` (registry helper), `gpu_vulkan_dispatcher.py` (Vulkan compute dispatcher), and `gpu_shaders/carrier_passthrough.comp` (sample GLSL) wire Vulkan kernels via dispatcher hooks without importing Vulkan into `dashi_core/`.
* PQ is optional and outside kernel semantics: `pq.py` provides 2-bit encode/decode helpers with tests under `tests/pq/`; kernels and adapters operate on dense `Carrier` only. Policy and guidance live in `docs/pq_policy.md`.
* Tests mirror theory under `tests/` (`carrier/`, `kernel/`, `defect/`, `admissibility/`, `mdl/`, `hierarchy/`, `backend/`, `violations/`, `reproducibility/`); run with `python -m pytest`.
* Dependencies: install with `python -m pip install -r requirements-dev.txt` (or `requirements.txt` for runtime only); no GPU/Vulkan deps are pulled into CORE.

## Quickstart (repo-wide usage)

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-dev.txt

# CPU-only tests
python -m pytest

# Optional: Vulkan FFT smoke (RADV example)
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json
python scripts/run_fft_smoke.py --fft-backend vkfft-vulkan
# Expected parity on RX 580: max reconstruction error ≈ 1e-6 for 256×256 complex64
```

### Enabling Vulkan VkFFT (optional)

1) Clone VkFFT headers (not vendored): `git clone https://github.com/DTolm/VkFFT.git VkFFT`  
2) Build the pybind11 bridge (needs Vulkan, glslang, spirv-tools headers/libs):

```bash
VKFFT_INCLUDE_DIR=VkFFT python setup_vkfft_vulkan.py bdist_wheel
python -m pip install dist/vkfft_vulkan_py-*.whl
```

3) Run the smoke test as above. The FFT adapter auto-falls back to NumPy on any probe or runtime error.

## Benchmarks (PQ vs dense)

Benchmarks live under `benchmarks/` and emit JSONL (not CI-gating). All outputs are timestamped automatically (`-<suite>-YYYYMMDD-HHMMSS.jsonl`).

### CPU (Intel i7-7700K, sign-flip kernel)

- **PQ roundtrip (encode+decode only):** for 65,536 elements, median ~55–58 ms vs dense copy baseline <1 ms. Packing overhead dominates at these sizes.
- **Kernel dense vs PQ:** for 65,536 elements, dense median ~0.7–3.0 ms (sparsity 0.0/0.5/0.9) vs PQ roundtrip + kernel ~58–61 ms. Dense path is faster across tested sizes (1k–65k) and sparsities.
- **PQ block-size sweep:** medians for 65,536 elements span ~55–123 ms depending on block size and sparsity. No block size beat the dense baseline in this sweep.

### GPU (RX 580, Vulkan, sign-flip kernel)

Dense Vulkan vs CPU reference (medians, ms):

| size | sparsity | CPU dense | Vulkan dense |
| ---- | -------- | --------- | ------------ |
| 1,024 | 0.0 | ~0.16 | ~6.54 |
| 1,024 | 0.5 | ~0.17 | ~6.38 |
| 1,024 | 0.9 | ~0.17 | ~6.07 |
| 16,384 | 0.0 | ~0.29 | ~8.43 |
| 16,384 | 0.5 | ~0.93 | ~8.48 |
| 16,384 | 0.9 | ~0.53 | ~7.78 |
| 65,536 | 0.0 | ~0.72 | ~9.92 |
| 65,536 | 0.5 | ~3.04 | ~12.02 |
| 65,536 | 0.9 | ~1.47 | ~10.32 |

All runs matched hashes (parity OK). Vulkan setup: RX 580 with RADV ICD (`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json`); SPIR-V built from `gpu_shaders/sign_flip.comp`.

Batch sweep (medians, ms) — dispatch repeated 1×/4×/8× per timing:

| size | sparsity | CPU 1× | CPU 4× | CPU 8× | Vulkan 1× | Vulkan 4× | Vulkan 8× |
| ---- | -------- | ------ | ------ | ------ | --------- | ---------- | ---------- |
| 1,024 | 0.0 | 0.19 | 0.51 | 1.56 | 7.49 | 25.98 | 51.96 |
| 1,024 | 0.5 | 0.30 | 1.16 | 1.45 | 6.17 | 30.95 | 59.47 |
| 1,024 | 0.9 | 0.24 | 0.93 | 1.39 | 6.73 | 23.59 | 54.18 |
| 16,384 | 0.0 | 0.41 | 1.22 | 2.11 | 8.39 | 29.67 | 54.48 |
| 16,384 | 0.5 | 1.24 | 3.49 | 8.27 | 9.06 | 33.03 | 66.73 |
| 16,384 | 0.9 | 0.44 | 1.81 | 4.04 | 7.19 | 32.68 | 64.48 |
| 65,536 | 0.0 | 0.68 | 2.72 | 7.00 | 10.65 | 39.59 | 79.73 |
| 65,536 | 0.5 | 4.38 | 17.07 | 23.97 | 13.24 | 53.85 | 110.68 |
| 65,536 | 0.9 | 1.39 | 8.74 | 13.43 | 11.01 | 45.88 | 89.07 |

Even with batching, the RX 580 remained slower than the i7-7700K on this light kernel; batching scales roughly linearly and does not close the gap for these sizes.

Run examples:
```
python benchmarks/bench.py --suite pq_roundtrip --sizes 1024 16384 65536 --sparsity 0.0 0.5 0.9 --repeats 5 --out benchmarks/results/pq_roundtrip.jsonl
python benchmarks/bench.py --suite kernel_dense_vs_pq --sizes 1024 16384 65536 --sparsity 0.0 0.5 0.9 --repeats 5 --out benchmarks/results/kernel_dense_vs_pq.jsonl
python benchmarks/bench.py --suite pq_block_sweep --sizes 1024 16384 65536 --sparsity 0.0 0.5 0.9 --blocks auto --repeats 3 --out benchmarks/results/pq_block_sweep.jsonl
export VK_ICD_FILENAMES=/path/to/radeon_icd.x86_64.json
glslc gpu_shaders/sign_flip.comp -o /tmp/sign_flip.spv
python benchmarks/bench.py --suite kernel_dense_vulkan --sizes 1024 16384 65536 --sparsity 0.0 0.5 0.9 --workload stencil_dense_iterated --iterations 10 --batches 1 4 8 --shader gpu_shaders/sign_flip.comp --spv /tmp/sign_flip.spv --device-index 0 --repeats 5 --out benchmarks/results/kernel_dense_vulkan.jsonl
```

Benchmark planning assets:
- Workload families and default axes: `benchmarks/workloads.py`
- Function coverage map: `benchmarks/coverage.yaml`
- Coverage guard: `python benchmarks/check_coverage.py`

### Backend notes (CPU vs GPU)

- These RX 580 runs confirm: for small/light kernels, Vulkan dispatch/transfer overhead dominates; batching amortizes linearly and does not overturn CPU > GPU on this hardware. This is expected and acceptable for dashiCORE, whose priority is semantic parity, not early GPU speedups.
- Guideline: if a GPU backend is slower than CPU for a given kernel, treat it as a performance observation, not a correctness failure. Use CPU as the semantic reference; expect GPU wins only for sufficiently large/dense workloads where overhead is amortized.
- For a GPU-favored regime, start with dense workloads (e.g., `--workload stencil_dense_iterated --iterations 10+` on large sizes) and an ALU-heavy single-pass workload (`--workload alu_dense_burn` with large sizes/rounds once added) to raise arithmetic intensity without changing semantics. Vulkan benchmarks now record `--memory-mode host_visible|device_local`; device-local mode uses staging to keep compute buffers in VRAM during timing (submit→fence), separating compute from PCIe/zero-copy overhead.

Latest dense, iterated sweep (RX 580, `stencil_dense_iterated`, `iterations=10`, `batches=4`, sparsity 0.0; medians, ms):

| size    | CPU | Vulkan |
| ------: | --: | -----: |
| 16,384  | 17.15 | 304.10 |
| 65,536  | 32.48 | 410.23 |
| 262,144 | 122.13 | 853.85 |

Parity remains perfect (hashes match); even at higher intensity the RX 580 stays slower than the i7-7700K. Treat this as a performance observation; correctness is validated.

Timing scope note (current runs): the Vulkan numbers above are end-to-end wall times with host-visible buffers and still include submission/synchronization cost. Next steps for a true GPU-favored crossover and compute isolation:

- Split timing into `submit_to_fence_ms` (dispatch→fence only) and `wall_ms` (includes staging/readback/hash); report both.
- Enforce a single fence wait per timed run (record N dispatches, submit once, wait once). Any per-dispatch fence wait should be treated as a bug and logged.
- Keep host-visible “zero-copy” mode as a separate, explicitly slower regime on discrete GPUs; report which mode is used.
- Make the ALU-heavy single-pass workload (`alu_dense_burn`) truly GPU-favored: rounds axis (e.g., 1 vs 4096) with ≥1M elements in device-local mode. A rounds sweep should show linear scaling if compute is being measured.

Latest Vulkan sweep with host-visible vs device-local staging (sign_flip shader):

| workload (batch=4) | size     | CPU | Vulkan host_visible | Vulkan device_local |
| ------------------ | -------: | --: | ------------------: | ------------------: |
| stencil_dense_iterated, iters=10, sparsity=0.0 | 65,536   | 29.66 | 383.25 | 411.24 |
| alu_dense_burn, iters=1, sparsity=0.0          | 1,048,576 | 56.68 | 224.18 | 244.81 |
| alu_dense_burn, iters=1, sparsity=0.0          | 4,194,304 | 245.37 | 884.80 | 867.47 |
| alu_dense_burn, iters=1, sparsity=0.0          | 16,777,216 | 1,042.57 | 3,741.79 | 3,673.29 |

Takeaways: device-local staging is wired but still dominated by submission/transfer overhead for these light kernels; GPU remains slower than CPU even for the larger ALU burn sizes. Compute-only timing and a heavier ALU shader remain the next levers to expose a GPU-favored crossover.

Latest GPU-only, device-local sanity run (RX 580, `alu_dense_burn`, size=1,048,576, batches=4, iterations=1):

- Config: GPU-only, RX 580 (RADV POLARIS10); workload `alu_dense_burn`; size 1,048,576; batches=4 (4 dispatches recorded per submit); iterations=1; memory_mode=device_local; fence_waits=1; hashes match.
- Compute timing: submit→fence ≈ 5.4–5.7 ms per run — VRAM-resident compute is fast and stable.
- Wall timing: 38–42 ms — about 34 ms overhead from staging/readback/hash/host bookkeeping.
- Outcome: GPU compute already beats prior CPU reference (~53 ms) for this config; remaining wall-time gap is host I/O, not dispatch or compute.
- Conclusion: GPU path is correct and compute-isolated; to drop wall time further you’d need GPU-side reduction or reduced readback scope, not Vulkan dispatch changes.

Latest GPU-only, device-local sanity run (RX 580, `alu_dense_burn`, size=1,048,576, batches=4, iterations=1):

- Config: GPU-only, RX 580 (RADV POLARIS10); workload `alu_dense_burn`; size 1,048,576; batches=4 (4 dispatches recorded per submit); iterations=1; memory_mode=device_local; fence_waits=1; hashes match.
- Compute timing: submit→fence ≈ 5.4–5.7 ms per run — VRAM-resident compute is fast and stable.
- Wall timing: 38–42 ms — about 34 ms overhead from staging/readback/hash/host bookkeeping.
- Outcome: GPU compute already beats prior CPU reference (~53 ms) for this config; remaining wall-time gap is host I/O, not dispatch or compute.
- Conclusion: GPU path is correct and compute-isolated; to drop wall time further you’d need GPU-side reduction or reduced readback scope, not Vulkan dispatch changes.
