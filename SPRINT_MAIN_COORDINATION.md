# dashiCORE — Initial Sprint Plan

> **Sprint theme:** *Formalise shape and IO. CPU-only. No optimisation.*
> **Primary artifact:** Stable, explicit mathematical interfaces.

---

## Sprint Objective

Establish a **canonical, minimal, CPU-based reference implementation** of the DASHI core mathematical objects, with:

* Explicit **data shapes**
* Explicit **input/output contracts**
* Explicit **invariants**

This sprint is considered successful when downstream projects can **import dashiCORE and rely on its interfaces**, even if they later replace implementations with GPU-accelerated versions.

---

## Non-Objectives (Explicit)

This sprint does **not** aim to:

* Achieve performance targets
* Support GPUs or accelerators
* Implement domain-specific kernels
* Tune heuristics or thresholds
* Handle large-scale datasets

Correctness, clarity, and stability dominate.

---

## Deliverable 1 — Canonical Data Shapes

### Goal

Define the **authoritative shapes** for all core mathematical objects.

These shapes must:

* Be unambiguous
* Be documented
* Be enforceable at runtime (assertions / type checks)

### Required Shapes

#### Carrier

* Signed ternary carrier:
  [
  T = {-1, 0, +1}
  ]
* Factorised as:

  * `support: Bool[...]`
  * `sign: Int8[...] ∈ {-1, +1}`

No representation that collapses these dimensions is allowed.

---

#### Kernel

* Input:

  * Carrier field (or structured container thereof)
* Output:

  * Same shape carrier field
* Optional:

  * Local neighborhood metadata
  * Admissibility context

Kernel IO must be **shape-preserving**.

---

#### Defect

* Scalar defect:

  * Per-site
  * Per-component
* Aggregated defect:

  * Normed
  * Decomposable

Defect shape must align with kernel domain.

---

#### Admissibility Context

* Explicit representation of:

  * Equivalence transformations
  * Gauge / redundancy actions
* Applied as:

  * A quotient map
  * Not a filter

---

## Deliverable 2 — Explicit IO Contracts

### Goal

Every public-facing object must declare:

* What it accepts
* What it returns
* What invariants it preserves

This is the **primary value** of dashiCORE.

---

### Required IO Guarantees

#### Kernel.apply

* **Input:** admissible carrier
* **Output:** admissible carrier
* **Guarantees:**

  * Shape preservation
  * No creation of invalid carrier values
  * Defect non-increase (or documented monotonicity)

---

#### Defect.compute

* **Input:** (pre, post) or single state
* **Output:** scalar + structured components
* **Guarantees:**

  * Zero iff fixed point
  * Invariant under admissibility transformations

---

#### MDL.score

* **Input:** representation
* **Output:** scalar complexity
* **Guarantees:**

  * Lower is preferred
  * Comparable across admissible representations

---

## Deliverable 3 — Naive CPU Reference Implementation

### Goal

Provide a **simple, readable, correct** implementation that:

* Mirrors the formal definitions
* Uses basic Python + NumPy
* Is intentionally slow

This implementation is the **gold standard for semantics**, not performance.

---

### Requirements

* No CUDA / ROCm / JAX / torch
* Deterministic behavior
* Minimal dependencies
* Exhaustive assertions in debug mode

GPU implementations elsewhere must match these semantics exactly.

---

## Deliverable 4 — Invariant & Contract Tests

### Goal

Encode the theory into tests.

If a downstream project violates a core invariant, the test should fail loudly.

---

### Required Tests

* Carrier validity (no illegal values)
* Support/sign separation
* Kernel idempotence (where applicable)
* Defect zero at fixed points
* Admissibility invariance
* Shape preservation across all public APIs

Tests are part of the API.

---

## Deliverable 5 — GPU Migration Boundary Definition

### Goal

Clearly define **what is allowed to change** when GPU acceleration is introduced later.

---

### Must Be Frozen

* Public class names
* IO shapes
* Mathematical semantics
* Defect definitions

---

### Allowed to Change

* Internal storage
* Execution backend
* Parallelism strategy
* Memory layout

This boundary must be documented explicitly.

---

## Sprint Milestones

### Milestone 1 — Skeleton & Types

* Directory structure
* Core dataclasses
* Shape assertions
* README finalized

---

### Milestone 2 — CPU Semantics

* Naive kernel application
* Defect computation
* Admissibility quotient handling

---

### Milestone 3 — Tests as Spec

* Contract tests
* Invariant enforcement
* Minimal example usage

---

## Sprint Exit Criteria (Hard)

The sprint is complete **only if**:

* All public APIs have documented IO shapes
* All invariants are enforced by code or tests
* A downstream project can import dashiCORE without redefining concepts
* A future GPU implementation can be validated against this CPU reference

---

## Sprint Mantra

> **If the shape is wrong, the theory is wrong.**
> **If the IO is unstable, nothing scales.**

