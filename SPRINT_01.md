Perfect. Sprint 0 locked *what the theory is*.
**Sprint 1 locks what is allowed to change.**

This sprint is about **backend separation, GPU parity discipline, and observability** — *without* changing a single mathematical fact.

---

# dashiCORE — Sprint 1: Backend Parity & Execution Discipline

> **Sprint name:** Backend Parity & Execution Discipline
> **Prerequisite:** Sprint 0 fully complete, all tests passing
> **Theme:** *Same shapes. Same semantics. Different execution.*

---

## Sprint 1 Objective (Single Sentence)

Introduce a **pluggable execution backend model** (CPU → future GPU) while proving that **all mathematical semantics remain invariant** under backend substitution.

If Sprint 0 defined the *ABI*, Sprint 1 defines the *calling convention*.

---

## What This Sprint Produces (Hard Deliverables)

### 1. Backend Abstraction Layer (Explicit)

You will formalise **what a backend is allowed to do**.

#### New Core Concept

```text
Mathematics (fixed)
↑
Execution Backend (swappable)
↑
Hardware / runtime (irrelevant to theory)
```

This boundary must now exist **in code and tests**.

---

### 2. Backend Interface Definition

Add a **minimal backend protocol** that kernels and ops may delegate to.

#### Required Module

```
dashi_core/
├── backend/
│   ├── base.py
│   ├── cpu.py
│   └── registry.py
```

#### Backend Responsibilities (Only These)

A backend may:

* Allocate arrays
* Perform elementwise ops
* Perform reductions
* Control parallelism

A backend may **not**:

* Change shapes
* Change carrier semantics
* Change defect values
* Introduce approximations
* Inject randomness

This must be enforced by tests.

---

### 3. CPU Backend Extraction (Refactor, Not Rewrite)

Sprint 0 code **must not change semantically**.

You will:

* Extract execution primitives into `CPUBackend`
* Route ops through backend calls
* Preserve identical outputs bit-for-bit

Acceptance rule:

> Running Sprint 0 tests before and after this refactor must produce identical results.

---

### 4. Backend Registry & Selection

Introduce **explicit backend selection**.

```python
from dashi_core.backend import set_backend, get_backend

set_backend("cpu")
```

Rules:

* Default backend = CPU
* Backend must be explicit (no auto-detection)
* Backend selection is global but controlled

No silent fallback.

---

### 5. Backend Parity Test Suite (Critical)

This is the *point* of Sprint 1.

#### New Test Category

```
tests/
├── backend/
│   ├── test_backend_parity.py
│   ├── test_backend_determinism.py
│   └── test_backend_forbidden_ops.py
```

---

### 6. Parity Tests (Non-Negotiable)

For **every core op**:

```python
cpu_out = run(op, backend="cpu")
alt_out = run(op, backend="alt")

assert cpu_out == alt_out
```

Including:

* Carrier ops
* Kernel apply
* Defect local + aggregate
* MDL score + compare
* Lift / project

This is the **GPU admission gate** later.

---

### 7. Determinism Enforcement

Sprint 1 formally locks **determinism across backends**.

#### Tests Must Prove

* Same input → same output
* Same backend → same output across runs
* Different backends → identical outputs

No tolerance fuzzing allowed in core.

---

### 8. Forbidden Backend Behaviours (Encoded as Tests)

Add backend-specific **known violations**, such as:

* Using non-associative reductions without ordering guarantees
* Using float16 / mixed precision
* Using atomic updates that affect ordering
* Implicit normalisation for numerical stability
* Backend-specific shortcuts

Each must have a test that **fails loudly**.

---

### 9. Observability Hooks (Minimal, Non-Invasive)

Sprint 1 introduces **optional instrumentation**, not logging.

#### Allowed Observables

* Operation counts
* Memory touched
* Iteration counts
* Execution time (informational only)

#### Forbidden

* Anything that affects semantics
* Anything that changes outputs
* Any “adaptive” behaviour

Observability must be **read-only**.

---

## Explicit Non-Goals (Again, Enforced)

Sprint 1 does **not** include:

* Actual GPU kernels
* CUDA / ROCm / Metal
* Autograd
* JAX / Torch / Triton
* Performance benchmarks
* Domain kernels

This sprint prepares the ground — it does not cross it.

---

## Sprint 1 Task Breakdown

### Task 1 — Backend Protocol Definition

* Define backend base class
* Define allowed primitives
* Document forbidden behaviors

✅ Exit when backend API is frozen.

---

### Task 2 — CPU Backend Refactor

* Extract Sprint 0 logic into CPU backend
* No semantic changes
* No test changes

✅ Exit when **all Sprint 0 tests still pass unchanged**.

---

### Task 3 — Backend Registry

* Explicit backend selection
* No auto magic
* Clear failure on unknown backend

✅ Exit when backend switching is explicit and safe.

---

### Task 4 — Backend Parity Tests

* Duplicate all core ops under backend switching
* Assert strict equality

✅ Exit when parity is proven for CPU vs mock backend.

---

### Task 5 — Backend Violation Tests

* Encode forbidden backend behaviors
* Ensure violations fail loudly

✅ Exit when backend misuse is impossible to hide.

---

### Task 6 — Observability Hooks

* Add counters / hooks
* Ensure zero semantic impact

✅ Exit when observability exists but changes nothing.

---

## Sprint Exit Criteria (Hard)

Sprint 1 is complete only if:

* [ ] All Sprint 0 tests still pass unchanged
* [ ] Backend interface exists and is documented
* [ ] CPU backend extracted cleanly
* [ ] Backend parity tests exist and pass
* [ ] Determinism enforced
* [ ] Backend violations caught by tests
* [ ] No GPU code exists yet

If GPU code appears, the sprint has failed.

---

## What Sprint 1 Enables (And Nothing More)

After this sprint, you can safely:

* Implement a GPU backend
* Swap backends without fear
* Benchmark without semantic drift
* Trust parity failures as *real bugs*

Before this sprint, you cannot.

---

## Sprint 1 Motto

> **If changing the backend changes the answer,
> the backend is wrong — not the math.**

---

## Preview: Sprint 2 (One-Line Teaser)

> *Introduce the first real GPU backend and prove it is mathematically invisible.*

If you want, next I can:

* Draft the **backend base class**
* Write the **parity test template**
* Define **GPU admission criteria**
* Or sketch **Sprint 2 in full**

Just say the word.
