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

Run examples:
```
python benchmarks/bench.py --suite pq_roundtrip --sizes 1024 16384 65536 --sparsity 0.0 0.5 0.9 --repeats 5 --out benchmarks/results/pq_roundtrip.jsonl
python benchmarks/bench.py --suite kernel_dense_vs_pq --sizes 1024 16384 65536 --sparsity 0.0 0.5 0.9 --repeats 5 --out benchmarks/results/kernel_dense_vs_pq.jsonl
python benchmarks/bench.py --suite pq_block_sweep --sizes 1024 16384 65536 --sparsity 0.0 0.5 0.9 --blocks auto --repeats 3 --out benchmarks/results/pq_block_sweep.jsonl
export VK_ICD_FILENAMES=/path/to/radeon_icd.x86_64.json
glslc gpu_shaders/sign_flip.comp -o /tmp/sign_flip.spv
python benchmarks/bench.py --suite kernel_dense_vulkan --sizes 1024 16384 65536 --sparsity 0.0 0.5 0.9 --shader gpu_shaders/sign_flip.comp --spv /tmp/sign_flip.spv --device-index 0 --repeats 5 --out benchmarks/results/kernel_dense_vulkan.jsonl
```
