Great — this is the sprint where dashiCORE **touches the real world** for the first time, but without surrendering an inch of theory.

Sprint 2 is where you **prove that acceleration is a detail**.

---

# dashiCORE — Sprint 2: First Accelerated Backend (GPU Admission)

> **Sprint name:** Accelerated Backend Admission
> **Prerequisite:** Sprint 1 complete, backend parity tests green
> **Theme:** *Make it fast without making it different.*

---

## Sprint 2 Objective (Single Sentence)

Introduce the **first non-CPU execution backend** (GPU or accelerator-backed) and prove—by construction and tests—that it is **mathematically invisible** to dashiCORE.

This sprint answers the question:

> “Can we accelerate without corrupting the kernel?”

---

## What This Sprint Produces (Hard Deliverables)

### 1. First Accelerated Backend (Real, Not Mock)

You will implement **one real accelerated backend**, chosen for practicality, not ideology.

Examples (choose exactly one):

* CUDA (via CuPy / raw kernels)
* ROCm
* Metal
* Vectorised CPU SIMD backend (acceptable if GPU infra not ready)

This backend must:

* Implement the backend protocol from Sprint 1
* Be selectable explicitly
* Pass **every existing test unchanged**

---

### 2. GPU Admission Gate (Formalised)

You will codify what it means for a backend to be “allowed”.

#### Admission Criteria (Hard)

A backend is admissible iff:

* ✅ All Sprint 0 tests pass
* ✅ All Sprint 1 parity tests pass
* ✅ Known violations still fail
* ✅ Determinism tests pass
* ✅ No new tolerance hacks introduced

This gate must be **written down** and enforced.

---

### 3. Precision Policy Lock (Critical)

Sprint 2 **locks numerical precision rules** forever.

#### Required Decisions

* Default precision (e.g. float32 vs float64)
* Whether mixed precision is allowed (default: ❌)
* Reduction order guarantees
* Explicit rejection of lossy fast-math flags

You must encode this in:

* Backend config
* Backend violation tests

---

### 4. Backend Capability Declaration

Each backend must now declare **what it supports**.

Example:

```python
BackendCapabilities(
    supports_float64=True,
    supports_atomic_ops=False,
    deterministic_reductions=True,
    supports_int8_exact=True,
)
```

Tests must fail if a kernel requires a capability the backend does not declare.

---

### 5. Performance *Measurement* (Not Optimisation)

Sprint 2 allows **measurement**, not tuning.

#### Allowed

* Wall-clock timing
* Operation counts
* Memory movement counts

#### Forbidden

* Changing algorithms
* Kernel fusion
* Approximate math
* Heuristic shortcuts

Performance is *observed*, not *pursued*.

---

### 6. Accelerated Backend Violation Tests

Add a new violation class:

```
tests/violations/test_backend_numeric_drift.py
```

These tests must catch:

* Float rounding drift
* Reduction-order nondeterminism
* Precision loss vs CPU
* Backend-specific “optimisations”

Failure message example:

> “Accelerated backend diverges from CPU semantics. This is forbidden.”

---

### 7. Cross-Backend Reproducibility Tests

New test category:

```
tests/reproducibility/
├── test_cross_backend_replay.py
```

Guarantee:

```python
cpu(x) == gpu(x) == cpu(x)  # after roundtrip
```

No “close enough”.

---

## Explicit Non-Goals (Still Enforced)

Sprint 2 does **not** include:

* Autograd
* Training loops
* Approximate kernels
* Domain kernels
* Heuristic pruning
* Memory-saving tricks
* Kernel fusion

Those belong to **downstream projects**, not dashiCORE.

---

## Sprint 2 Task Breakdown

### Task 1 — Choose and Bootstrap Backend

* Select GPU/accelerated backend
* Minimal viable implementation
* Explicit backend registration

✅ Exit when backend runs a trivial op.

---

### Task 2 — Backend Capability Declaration

* Define capability schema
* Enforce capability checks
* Add tests for missing capabilities

✅ Exit when unsupported ops fail loudly.

---

### Task 3 — Precision Lock

* Enforce precision choices
* Ban mixed precision
* Add numeric drift tests

✅ Exit when CPU vs GPU matches bit-for-bit (where required).

---

### Task 4 — Parity Enforcement

* Run full test suite on accelerated backend
* Fix all divergences

✅ Exit when **no test distinguishes CPU from GPU**.

---

### Task 5 — Observability on Accelerated Backend

* Enable performance counters
* Ensure observability is read-only

✅ Exit when metrics exist but outputs unchanged.

---

## Sprint Exit Criteria (Non-Negotiable)

Sprint 2 is complete only if:

* [ ] Accelerated backend exists and is selectable
* [ ] All existing tests pass unchanged
* [ ] Backend admission criteria documented
* [ ] Numeric drift tests pass
* [ ] Determinism preserved
* [ ] No algorithmic changes introduced
* [ ] CPU remains the semantic reference

If you have to loosen a test to “make GPU work”, the sprint has failed.

---

## What Sprint 2 Unlocks (Safely)

After this sprint, you may:

* Trust GPU results
* Scale problem sizes
* Benchmark performance meaningfully
* Build domain kernels downstream

Before this sprint, GPU use is speculative.

---

## Sprint 2 Motto

> **Acceleration is allowed.
> Approximation is not.**

---

## Preview: Sprint 3 (Optional, Strategic)

> *Introduce domain-facing kernel interfaces and prove they cannot violate core invariants.*

If you want next, I can:

* Draft the **GPU backend skeleton**
* Write the **numeric drift test templates**
* Help choose **CUDA vs ROCm vs Metal** strategically
* Or fully define **Sprint 3**

Just tell me where you want to go next.
