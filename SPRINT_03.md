This is exactly the *right* way to test whether dashiCORE is real or just tidy theory.

What you’re proposing is **not** “add a GPU backend”, but:

> **Prove that dashiCORE can wrap an *already-existing*, opinionated GPU implementation without contaminating CORE.**

That’s a much stronger modularity test than writing a fresh backend.

Below is a **cleanly defined next sprint** that treats the external GPU project as a *hostile but useful artifact*.

---

# dashiCORE — Sprint 3: External GPU Backend Integration (Modularity Proof)

> **Sprint name:** External Backend Integration & Modularity Proof
> **Prerequisite:** Sprint 2 complete (accelerated backend admitted)
> **Theme:** *CORE adapts. CORE does not bend.*

---

## Sprint 3 Objective (Single Sentence)

Demonstrate that **dashiCORE can impose its semantics unchanged** on an existing, independently-developed GPU implementation, thereby proving that CORE is **truly modular and not co-designed with its backends**.

If Sprint 2 proved “we can write a GPU backend,”
Sprint 3 proves “we don’t have to.”

---

## Framing (Important)

The external GPU project:

* Was **not designed for dashiCORE**
* Has its own assumptions
* Likely violates multiple CORE invariants internally
* May use:

  * Different carriers
  * Different memory layouts
  * Different reduction semantics
  * Different normalisations

**Sprint 3 succeeds only if CORE remains unchanged.**

Any required change to CORE is considered a **failure**, not a refactor.

---

## What This Sprint Produces (Hard Deliverables)

### 1. External Backend Adapter Layer

You will introduce a **strict adapter**, not a rewrite.

#### New Module

```
dashi_core/
├── backend/
│   └── adapters/
│       └── external_gpu_project.py
```

This adapter:

* Translates CORE ops → external GPU calls
* Translates external outputs → CORE carriers
* Enforces CORE invariants at the boundary
* Performs validation before and after every call

The external project is treated as **unsafe until proven otherwise**.

---

### 2. Explicit Boundary Mapping Document

You must write (and commit) a short document:

```
docs/external_backend_mapping.md
```

It answers:

| Question                                  | Required Answer     |
| ----------------------------------------- | ------------------- |
| What does CORE provide?                   | Exact ops + shapes  |
| What does external project provide?       | GPU primitives      |
| What is translated?                       | Data, not semantics |
| What is rejected?                         | Any semantic drift  |
| What invariants are enforced at boundary? | Listed explicitly   |

This document is **part of the sprint acceptance**.

---

### 3. Adapter-Level Violation Catching

You will deliberately test that the adapter:

* Detects and rejects:

  * Non-ternary carriers
  * Implicit normalisation
  * Support creation
  * Shape drift
  * Approximate reductions
* Raises CORE-style errors, not backend errors

This proves CORE remains sovereign.

---

### 4. Cross-Implementation Parity Tests (Strongest Yet)

Add a new parity category:

```
tests/parity/
├── test_cpu_vs_external_gpu.py
```

For all mock kernels and at least one nontrivial pipeline:

```python
cpu_out = run(core_pipeline, backend="cpu")
ext_out = run(core_pipeline, backend="external_gpu")

assert cpu_out == ext_out
```

No tolerance widening allowed.

---

### 5. Modularity Stress Tests (Adversarial)

Add tests that intentionally expose mismatch risks:

* External backend uses:

  * float32 only
  * different reduction ordering
  * fused kernels
* Adapter must:

  * reject unsupported modes
  * or force safe execution paths

These tests **must fail** if invariants are violated.

---

### 6. No CORE Changes Rule (Hard)

During Sprint 3:

* ❌ No changes to:

  * Carrier semantics
  * OPS / IO table
  * Defect definition
  * MDL definition
  * Test expectations
* ✅ Only allowed changes:

  * Adapter code
  * Adapter tests
  * Documentation

If CORE must change to “support” the external backend, the sprint fails.

---

## Sprint 3 Task Breakdown

### Task 1 — Select External GPU Project

Choose **one existing project** of yours that:

* Is GPU-accelerated
* Has nontrivial logic
* Was not designed with dashiCORE in mind

Document its assumptions explicitly.

✅ Exit when assumptions are written down.

---

### Task 2 — Define Adapter Contract

Write a minimal adapter interface:

```python
class ExternalGPUAdapter(Backend):
    def apply_kernel(...)
    def compute_defect(...)
```

No leakage of external concepts into CORE.

✅ Exit when adapter API mirrors CORE backend API exactly.

---

### Task 3 — Boundary Enforcement

Implement:

* Pre-call validation (inputs)
* Post-call validation (outputs)
* Shape + carrier checks

✅ Exit when violations are caught immediately.

---

### Task 4 — Parity & Violation Tests

* Add CPU vs external backend parity tests
* Add explicit failure tests for known external violations

✅ Exit when parity passes and violations fail.

---

### Task 5 — Documentation Lock

Write:

* `external_backend_mapping.md`
* A short README section:

> “CORE does not adapt to backends.
> Backends adapt to CORE.”

✅ Exit when documentation explains the philosophy clearly.

---

## Sprint Exit Criteria (Non-Negotiable)

Sprint 3 is complete only if:

* [ ] External GPU backend runs via adapter
* [ ] CORE code unchanged
* [ ] All existing tests pass unchanged
* [ ] New parity tests pass
* [ ] External backend violations are detected
* [ ] Boundary mapping documented
* [ ] Adapter is optional and isolated

If any CORE test is weakened, the sprint has failed.

---

## What Sprint 3 Proves (This Is the Point)

After this sprint, you can say—truthfully:

> “dashiCORE is not coupled to any execution model.
> It can impose its semantics on pre-existing systems.”

That’s a *very* strong claim, and very few frameworks can make it honestly.

---

## Sprint 3 Motto

> **If CORE bends to fit the backend,
> then CORE was never a core.**

---

## Preview: Sprint 4 (Optional, Strategic)

> *Expose a formal ‘CORE compliance’ test kit for external projects.*

This would let any project certify itself as “dashiCORE-compatible” without merging code.

If you want, next I can:

* Help you **choose which GPU project** is the best stress test
* Draft the **adapter skeleton**
* Write a **CORE compliance checklist**
* Or define **Sprint 4** in detail

Just tell me where you want to go next.
