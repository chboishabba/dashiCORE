Alright — here’s a **tight, executable First Sprint definition** you can literally paste into `SPRINT.md` and start coding against. This is the sprint where **nothing clever is allowed** and **everything is explicit**.

---

# dashiCORE — Sprint 0: Core Shape Lock

> **Sprint name:** Core Shape & Semantics Lock
> **Duration:** 1 short sprint (time-boxed; stop when exit criteria met)
> **Theme:** *Formalise shape and IO. CPU-only. Tests define truth.*

---

## Sprint Objective (Single Sentence)

Lock the **canonical mathematical interface** of dashiCORE by implementing a naive CPU reference whose **shapes, IO contracts, and invariants are enforced by tests**.

If this sprint succeeds, *everything else becomes replaceable*.

---

## What This Sprint Produces (Hard Deliverables)

### 1. Canonical Core Types (No Algorithms Yet)

You will implement **only the minimal datatypes required** to express the theory.

#### Required Modules

```
dashi_core/
├── carrier.py
├── kernel.py
├── defect.py
├── admissibility.py
├── mdl.py
└── hierarchy.py
```

Each module must contain:

* A single core abstraction
* Explicit invariants
* No domain logic
* No performance optimisations

---

### 2. Explicit OPS / IO Compliance

All public methods must conform to the **OPS / IO table** you already defined.

**Acceptance rule:**

> If a method exists, it must be in the OPS table.
> If it’s in the OPS table, it must exist.

No extras. No shortcuts.

---

### 3. Naive CPU Reference Semantics

* Python + NumPy only
* Deterministic
* Intentionally slow
* Readable enough to serve as executable documentation

This implementation is **the semantic gold standard**.

---

### 4. Minimal Mock Kernels

Implement **exactly** the agreed mock kernels:

* `IdentityKernel`
* `ZeroKernel`
* `ClampKernel`
* `OneStepErodeKernel`
* `AdmissibilityNeutralKernel`

These live under:

```
dashi_core/testing/mock_kernels.py
```

They exist **only** to exercise invariants.

---

### 5. Full Initial Test Suite

All of the following must exist and pass:

#### Spec Tests

* Carrier tests
* Kernel tests
* Defect tests
* Admissibility tests
* MDL tests
* Hierarchy tests

#### Known Violations Suite

* Support/sign collapse
* Support creation
* Shape drift
* Defect-as-loss misuse
* Gauge dependence
* Hidden normalisation
* Illegal carrier states
* Hierarchy leakage

Tests are **normative**, not illustrative.

---

## Explicit Non-Goals (Enforced)

This sprint must **not** include:

* GPU code
* Parallelism
* Autograd
* ML frameworks
* Heuristics
* Optimisation
* Domain kernels
* Visualization
* IO / file formats

If it feels useful, stop and remove it.

---

## Sprint Tasks (Ordered)

### Task 1 — Repo Skeleton & README Lock

* Add README with purpose + non-goals
* Add OPS / IO table
* Add sprint doctrine

✅ Exit when README alone explains the theory.

---

### Task 2 — Carrier Implementation

* `from_signed`
* `to_signed`
* `validate`
* Shape enforcement
* Fail-fast on illegal states

✅ Exit when **carrier tests + violations pass**.

---

### Task 3 — Kernel Base + Mock Kernels

* Kernel base class
* Mock kernels implemented
* Explicit `is_idempotent` declaration

✅ Exit when **kernel tests pass using only mocks**.

---

### Task 4 — Defect Semantics

* Local defect
* Aggregation
* Zero-test

✅ Exit when **defect is zero iff fixed point**.

---

### Task 5 — Admissibility Quotient

* Equivalence relation
* Transform application
* Invariance guarantees

✅ Exit when **gauge tests pass**.

---

### Task 6 — MDL Skeleton

* Score
* Compare
* Invariance under admissibility

✅ Exit when **ordering is deterministic and invariant**.

---

### Task 7 — Hierarchy Skeleton

* Lift
* Project
* Explicit shape change
* No implicit leakage

✅ Exit when **hierarchy roundtrip tests pass**.

---

### Task 8 — Known Violations Lock

* All violation tests implemented
* All violations fail loudly
* Error messages educational

✅ Exit when **every historical mistake is guarded**.

---

## Sprint Exit Criteria (Non-Negotiable)

The sprint is complete **only if**:

* [ ] All OPS / IO table entries are implemented
* [ ] All tests pass on naive CPU
* [ ] All known violations fail loudly
* [ ] No public API lacks a test
* [ ] Shapes are explicit everywhere
* [ ] No domain logic exists
* [ ] No performance code exists

If any box is unchecked, the sprint is not done.

---

## What You Do *After* This Sprint (But Not Now)

* GPU backends
* Domain kernels
* Performance tuning
* Heuristics
* Extensions

Those are **Sprint 1+ concerns**.

---

## Sprint Motto (Worth Writing at the Top of the Board)

> **This sprint does not solve problems.
> It defines what a solution is allowed to look like.**

---

If you want, next we can:

* Break this sprint into **daily commit checkpoints**
* Generate a **task-by-task commit message template**
* Draft **Sprint 1 (GPU parity & backend swap)**
* Or write the **first failing test** to kick things off properly

Just say where you want to begin.
