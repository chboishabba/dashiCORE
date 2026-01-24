# dashiCORE — Initial Test Specification

> **Principle:**
> If a test fails, the theory is violated — not just the implementation.

---

## Test Categories (Top-Level)

```
tests/
├── carrier/
├── kernel/
├── defect/
├── admissibility/
├── mdl/
├── hierarchy/
└── integration/
```

Each category enforces **one conceptual layer**.

---

## 1. Carrier Tests

### 1.1 `test_carrier_from_signed_roundtrip`

**Purpose:** Enforce canonical support × sign factorisation.

**Given**

```python
signed = [-1, 0, +1, +1, 0, -1]
```

**Test**

* `Carrier.from_signed(signed).to_signed() == signed`

**Invariants**

* No information loss
* Exact inverse

---

### 1.2 `test_carrier_support_sign_separation`

**Purpose:** Prevent conflation of existence and orientation.

**Assertions**

* `support == (signed != 0)`
* `sign ∈ {-1,+1}` wherever `support == True`
* `sign` ignored wherever `support == False`

---

### 1.3 `test_carrier_rejects_illegal_values`

**Purpose:** Guard against silent corruption.

**Given**

```python
signed = [2, -2, 42]
```

**Expect**

* Exception raised
* No partial construction

---

### 1.4 `test_carrier_shape_preservation`

**Purpose:** Shape is sacred.

**Given**

* Arbitrary shape `Ω` (e.g. `(4,3,2)`)

**Assert**

* `support.shape == Ω`
* `sign.shape == Ω`

---

## 2. Kernel Tests

### 2.1 `test_kernel_shape_preserving`

**Purpose:** Kernels may change values, never shapes.

**Assert**

```python
kernel.apply(state).shape == state.shape
```

---

### 2.2 `test_kernel_valid_output`

**Purpose:** Kernel must not emit illegal carrier states.

**Assert**

* Output passes `Carrier.validate()`

---

### 2.3 `test_kernel_fixed_point_zero_defect`

**Purpose:** Definition of consistency.

**Given**

```python
x = kernel.apply(x)
```

**Assert**

```python
Defect.is_zero(x, kernel) == True
```

---

### 2.4 `test_kernel_idempotence_if_declared`

**Purpose:** Enforce declared algebraic properties.

**If**

```python
kernel.is_idempotent == True
```

**Assert**

```python
kernel.apply(kernel.apply(x)) == kernel.apply(x)
```

---

## 3. Defect Tests

### 3.1 `test_defect_non_negative`

**Purpose:** Defect is a measure, not a signal.

**Assert**

```python
Defect.local(pre, post) >= 0
Defect.aggregate(local) >= 0
```

---

### 3.2 `test_defect_zero_iff_fixed_point`

**Purpose:** Formal definition of defect.

**Assert**

```python
Defect.aggregate(Defect.local(x, kernel.apply(x))) == 0
⇔
kernel.apply(x) == x
```

---

### 3.3 `test_defect_shape_alignment`

**Purpose:** Prevent misaligned diagnostics.

**Assert**

```python
Defect.local(pre, post).shape == pre.shape
```

---

### 3.4 `test_defect_monotonicity_under_kernel`

**Purpose:** Contractivity requirement.

**Assert**

```python
Defect.aggregate(local(x, K(x))) >=
Defect.aggregate(local(K(x), K(K(x))))
```

(Or documented alternative if kernel is non-contractive.)

---

## 4. Admissibility Tests

### 4.1 `test_admissibility_invariance_of_defect`

**Purpose:** Gauge ≠ physics.

**Assert**

```python
Defect.aggregate(x, K(x)) ==
Defect.aggregate(A(x), K(A(x)))
```

---

### 4.2 `test_admissibility_equivalence_relation`

**Purpose:** Quotient well-defined.

**Assert**

* Reflexive: `A.equivalent(x, x)`
* Symmetric
* Transitive

---

### 4.3 `test_admissibility_preserves_shape`

**Purpose:** Transformations are structure-preserving.

**Assert**

```python
A(x).shape == x.shape
```

---

## 5. MDL Tests

### 5.1 `test_mdl_non_negative`

**Purpose:** Complexity is a measure.

**Assert**

```python
MDL.score(rep) >= 0
```

---

### 5.2 `test_mdl_invariance_under_admissibility`

**Purpose:** Representation-independent preference.

**Assert**

```python
MDL.score(x) == MDL.score(A(x))
```

---

### 5.3 `test_mdl_comparison_total_order`

**Purpose:** MDL must break ties deterministically or declare ties.

**Assert**

```python
MDL.compare(a, b) ∈ {A, B, tie}
```

---

## 6. Hierarchy (M-Level) Tests

### 6.1 `test_lift_project_roundtrip_no_spurious_defect`

**Purpose:** Hierarchy must compose cleanly.

**Assert**

```python
Project(Lift(x)) ≈ x
```

(Up to admissibility equivalence.)

---

### 6.2 `test_hierarchy_defect_composition`

**Purpose:** Defects must scale coherently.

**Assert**

```python
Defect_Mk(Lift(x)) == f(Defect_Mn(x))
```

for declared composition rule `f`.

---

## 7. Integration Tests (Minimal)

### 7.1 `test_cpu_reference_pipeline`

**Purpose:** Lock semantics end-to-end.

**Pipeline**

```python
x0 → Kernel → Defect → Admissibility → MDL
```

**Assert**

* No invariant violations
* Deterministic output
* Shape preserved throughout

---

### 7.2 `test_backend_equivalence_placeholder`

**Purpose:** Future GPU parity hook.

**Assert**

```python
cpu_impl(x) == backend_impl(x)
```

(Currently backend_impl = cpu_impl.)

---

## 8. Forbidden Failure Modes (Tests MUST Catch)

| Failure                      | Must Be Caught By      |
| ---------------------------- | ---------------------- |
| Silent shape change          | Carrier / Kernel tests |
| Support/sign collapse        | Carrier tests          |
| Defect < 0                   | Defect tests           |
| Gauge-dependent behavior     | Admissibility tests    |
| MDL changing under transform | MDL tests              |
| Hierarchy creating defect    | Hierarchy tests        |

---

## 9. Exit Criteria for Initial Test Suite

This test set is complete when:

* All tests pass on naive CPU implementation
* Every public op is exercised at least once
* No test depends on domain semantics
* A GPU backend can be dropped in and validated by **running the same tests**

---

## One-Line Philosophy (Put This in `tests/README.md`)

> **These tests do not check behavior.
> They define what behavior is allowed.**



# dashiCORE — Minimal Mock Kernels

> **Purpose:**
> Provide mathematically trivial kernels that nevertheless fully exercise:
>
> * carrier validity
> * shape preservation
> * defect semantics
> * idempotence vs contractivity
> * admissibility invariance

No domain knowledge. No heuristics. No learning.

---

## Design Rules for Mock Kernels

All mock kernels MUST:

1. Preserve shape exactly
2. Produce valid `Carrier` outputs
3. Be deterministic
4. Have fully understood fixed points
5. Declare algebraic properties explicitly

If a kernel feels “useful”, it does not belong here.

---

## Base Interface (Assumed)

```python
class Kernel:
    is_idempotent: bool

    def apply(self, state: Carrier, ctx=None) -> Carrier:
        ...
```

---

## 1. IdentityKernel

### Purpose

Baseline sanity check. Defines **zero defect everywhere**.

### Definition

```python
class IdentityKernel(Kernel):
    is_idempotent = True

    def apply(self, state, ctx=None):
        return state
```

### Properties

| Property         | Value      |
| ---------------- | ---------- |
| Shape-preserving | ✅          |
| Creates support  | ❌          |
| Idempotent       | ✅          |
| Contractive      | Trivial    |
| Fixed points     | All states |

### Tests Enabled

* `test_kernel_fixed_point_zero_defect`
* `test_defect_zero_iff_fixed_point`
* `test_admissibility_invariance_of_defect`
* `test_kernel_idempotence_if_declared`

This kernel is **the zero object** in kernel space.

---

## 2. ZeroKernel (Hard Projector)

### Purpose

Tests **support elimination**, defect reduction, and projection behavior.

### Definition

```python
class ZeroKernel(Kernel):
    is_idempotent = True

    def apply(self, state, ctx=None):
        support = np.zeros_like(state.support, dtype=bool)
        sign = np.ones_like(state.sign, dtype=np.int8)  # ignored
        return Carrier(support=support, sign=sign)
```

### Properties

| Property         | Value              |
| ---------------- | ------------------ |
| Shape-preserving | ✅                  |
| Creates support  | ❌                  |
| Destroys support | ✅                  |
| Idempotent       | ✅                  |
| Fixed points     | Zero-support state |

### Notes

* Sign is meaningless when support is false
* Tests support/sign separation hard

### Tests Enabled

* `test_carrier_support_sign_separation`
* `test_kernel_valid_output`
* `test_defect_monotonicity_under_kernel`

---

## 3. ClampKernel (Sign Normaliser)

### Purpose

Tests **sign correction without support change**.

### Definition

```python
class ClampKernel(Kernel):
    is_idempotent = True

    def apply(self, state, ctx=None):
        sign = np.where(state.support, np.sign(state.sign), state.sign)
        sign = np.where(sign == 0, 1, sign)  # force ±1
        return Carrier(state.support, sign.astype(np.int8))
```

### Properties

| Property         | Value                  |
| ---------------- | ---------------------- |
| Shape-preserving | ✅                      |
| Changes support  | ❌                      |
| Changes sign     | ✅                      |
| Idempotent       | ✅                      |
| Fixed points     | Properly signed states |

### Tests Enabled

* `test_carrier_validate`
* `test_kernel_valid_output`
* `test_defect_localisation`

This kernel catches **illegal sign leakage**.

---

## 4. OneStepErodeKernel (Contractive, Non-Idempotent)

### Purpose

Test **defect monotonicity over iterations**.

### Definition (example erosion)

```python
class OneStepErodeKernel(Kernel):
    is_idempotent = False

    def apply(self, state, ctx=None):
        support = state.support.copy()
        # drop one arbitrary supported site (deterministic choice)
        idx = np.argwhere(support)
        if len(idx) > 0:
            support[tuple(idx[0])] = False
        return Carrier(support, state.sign)
```

### Properties

| Property         | Value              |
| ---------------- | ------------------ |
| Shape-preserving | ✅                  |
| Idempotent       | ❌                  |
| Contractive      | ✅                  |
| Fixed points     | Zero-support state |

### Tests Enabled

* `test_defect_monotonicity_under_kernel`
* `test_kernel_non_idempotence`
* `test_iterated_convergence`

This kernel proves your defect logic actually tracks progress.

---

## 5. AdmissibilityNeutralKernel

### Purpose

Explicitly test **gauge invariance**.

### Definition

```python
class AdmissibilityNeutralKernel(Kernel):
    is_idempotent = True

    def apply(self, state, ctx=None):
        # Ignores coordinate order, acts pointwise
        return Carrier(state.support, state.sign)
```

Used in combination with **nontrivial admissibility transforms**.

### Tests Enabled

* `test_admissibility_invariance_of_defect`
* `test_mdl_invariance_under_admissibility`

---

## 6. Forbidden Mock (Documented Anti-Example)

> **Do NOT implement — only document**

### IllegalKernel (Support Creation)

```python
# This must NEVER exist
support = np.ones_like(state.support)
```

Why forbidden:

* Violates kernel contract
* Breaks MDL monotonicity
* Destroys interpretability

Include as **commented example** in `tests/README.md`.

---

## Minimal Set Summary

| Kernel                     | Idempotent | Contractive | Primary Purpose      |
| -------------------------- | ---------- | ----------- | -------------------- |
| IdentityKernel             | ✅          | trivial     | Zero baseline        |
| ZeroKernel                 | ✅          | ✅           | Projection semantics |
| ClampKernel                | ✅          | trivial     | Carrier validity     |
| OneStepErodeKernel         | ❌          | ✅           | Defect dynamics      |
| AdmissibilityNeutralKernel | ✅          | trivial     | Gauge invariance     |

You only need **these five** to lock the core.

---

## Where These Live

Suggested layout:

```
dashi_core/
├── kernel/
│   └── base.py
└── testing/
    └── mock_kernels.py
```

Downstream projects **must never depend on these**, only the tests.

---

## One-Line Rule (Worth Writing in Code)

```python
# Mock kernels are not examples of use.
# They are witnesses to invariants.
```


# dashiCORE — Known Violations Test Suite

> **Purpose:**
> Encode *historically common, theoretically invalid patterns* as tests that must fail.
> These tests protect dashiCORE from regression and downstream misuse.

> **Rule:**
> If a downstream project triggers one of these failures, the project is wrong — not the test.

---

## Structure

```
tests/
├── violations/
│   ├── test_support_sign_collapse.py
│   ├── test_support_creation.py
│   ├── test_shape_drift.py
│   ├── test_defect_as_loss.py
│   ├── test_gauge_dependent_behavior.py
│   ├── test_hidden_normalisation.py
│   ├── test_illegal_carrier_states.py
│   └── test_hierarchy_leakage.py
```

Each file documents **one class of mistake**.

---

## 1. Support / Sign Collapse

### Violation

Treating the carrier as a real-valued field (`[-1,1]`, `[0,1]`, logits, probabilities).

### Why It’s Wrong

* Conflates existence with orientation
* Breaks involution symmetry
* Destroys MDL optimality
* Makes defect ill-defined

### Test: `test_support_sign_collapse_rejected`

**Given**

```python
bad = np.array([0.2, -0.7, 0.0])
```

**Expect**

* `Carrier.from_signed(bad)` raises
* Any kernel accepting this fails validation

**Failure Message**

> “Carrier values must be ternary. Probabilistic mass is not a carrier.”

---

## 2. Support Creation by Kernels

### Violation

Kernel creates new support where none existed.

### Why It’s Wrong

* Invents structure
* Violates projection semantics
* Breaks MDL monotonicity

### Test: `test_kernel_must_not_create_support`

**Given**

```python
state.support == False everywhere
```

**Assert**

```python
kernel.apply(state).support.sum() == 0
```

**Failure Message**

> “Kernel created support from nothing. This is forbidden.”

---

## 3. Shape Drift

### Violation

Kernel or defect changes tensor shape silently.

### Why It’s Wrong

* Breaks composability
* Makes hierarchy ill-defined
* Breaks GPU parity

### Test: `test_shape_drift_detected`

**Assert**

```python
assert out.shape == in.shape
```

**Failure Message**

> “Shape drift detected. Shapes are part of the theory.”

---

## 4. Treating Defect as a Loss Function

### Violation

Using defect as:

* A gradient signal
* A soft penalty
* Something to be “minimised” via optimisation

### Why It’s Wrong

* Defect is a **diagnostic**, not an objective
* Kernel defines correction, not gradient descent
* Leads to non-invariant behaviour

### Test: `test_defect_not_used_as_loss`

**Check**

* Defect computation must not mutate state
* No gradients, no parameter updates

**Failure Message**

> “Defect is not a loss. Optimisation here is category error.”

---

## 5. Gauge-Dependent Behaviour

### Violation

Results differ under admissibility transformations.

### Why It’s Wrong

* Coordinates are gauge
* Only invariants survive
* Anything else is artefact

### Test: `test_gauge_dependence_detected`

**Assert**

```python
kernel(x) ≡ kernel(A(x))
```

(up to admissibility equivalence)

**Failure Message**

> “Kernel output depends on gauge. This is illegal.”

---

## 6. Hidden Normalisation / Renormalisation

### Violation

Implicitly rescaling, clipping, or normalising values.

Examples:

* Normalising magnitudes
* Enforcing sum-to-one
* Auto-scaling fields

### Why It’s Wrong

* Breaks carrier semantics
* Injects untracked assumptions
* Destroys interpretability

### Test: `test_hidden_normalisation_forbidden`

**Detect**

* Any change in sign/support counts without kernel justification

**Failure Message**

> “Implicit normalisation detected. All transformations must be explicit.”

---

## 7. Illegal Carrier States After Ops

### Violation

Kernel or transform outputs:

* `sign == 0`
* `sign != ±1` where support is true
* Non-boolean support

### Why It’s Wrong

* Corrupts carrier
* Makes defect meaningless

### Test: `test_illegal_carrier_state_emitted`

**Assert**

```python
Carrier.validate(out)
```

**Failure Message**

> “Operation emitted illegal carrier state.”

---

## 8. Hierarchy Leakage

### Violation

Lower-level structure leaks upward or downward without explicit lift/project.

Examples:

* M3 details implicitly appearing in M6
* Higher-order effects injected without lift

### Why It’s Wrong

* Breaks compositionality
* Makes hierarchy non-functorial
* Destroys reasoning across scales

### Test: `test_hierarchy_requires_explicit_lift`

**Assert**

```python
Lift(x) required before Mk ops
```

**Failure Message**

> “Hierarchy leakage detected. Use explicit lift/project.”

---

## 9. Meta-Test: Violations Must Fail

### Test: `test_known_violations_fail_loudly`

**Purpose**
Ensure these tests actually catch errors.

**Assert**

* Each violation raises
* Error messages are explicit and educational

---

## Philosophy Block (Put in `tests/violations/README.md`)

> These tests exist because:
>
> * Every one of these mistakes has already happened
> * Every one of them looked reasonable at the time
> * Every one of them caused subtle, expensive breakage later
>
> If you think you need to bypass one of these tests,
> you are probably redefining the theory.

---

## How Downstream Projects Use This

Downstream repos should:

* Import and run `tests/violations` unchanged
* Treat failures as **design errors**
* Never monkey-patch these tests away

---

## Exit Criteria for This Suite

This suite is complete when:

* Every historically observed misuse has a test
* Each test fails with a *clear explanation*
* No violation test depends on domain logic
* GPU backends fail the same violations as CPU


# dashiCORE — Pre-Start Considerations

> **If it isn’t decided here, it will be decided accidentally later.**

---

## 1. What Is *Normative* vs *Illustrative*

### Decision to Lock Now

Which parts of the repo are **the law**, and which are **examples**?

### Recommendation

Declare explicitly:

* **Normative**

  * Core ops / IO spec
  * Carrier, Kernel, Defect semantics
  * Tests (especially violations)
* **Illustrative only**

  * Mock kernels
  * Example pipelines
  * Comments with intuition

Put this sentence in the README:

> “Tests define semantics. Examples do not.”

This avoids downstream cargo-culting mock logic.

---

## 2. Equality vs Equivalence (Critical)

You already *use* this distinction implicitly. Lock it explicitly.

### Questions to Answer Now

* When do we require **exact equality**?
* When do we allow **admissibility equivalence**?

### Recommendation

Define three relations:

| Name              | Meaning                     | Used For                   |
| ----------------- | --------------------------- | -------------------------- |
| `==`              | Bitwise equality            | Shapes, carriers, tests    |
| `equivalent(a,b)` | Same admissible class       | Kernel/admissibility tests |
| `≈`               | Equal up to projection/lift | Hierarchy roundtrips       |

If you don’t formalise this now, GPU parity will be impossible.

---

## 3. Error Policy (Fail Fast vs Tolerant)

### Question

Do illegal states:

* raise immediately?
* auto-correct?
* log and continue?

### Recommendation (Strong)

**Fail fast, no recovery** in dashiCORE.

* Illegal carrier → exception
* Shape mismatch → exception
* Gauge-dependent behavior → exception

Downstream projects can soften this. Core must not.

---

## 4. Determinism Guarantees

### Question

Is every operation required to be deterministic?

### Recommendation

Yes — **explicitly**.

Even mock kernels must:

* Choose erosion sites deterministically
* Avoid iteration order dependence
* Avoid hash-based randomness

Write this down now, or tests will flake later.

---

## 5. Floating-Point Policy

This one bites late if ignored.

### Questions

* Are floats allowed at all?
* If yes, where?
* With what tolerance?

### Recommendation

* **Carrier:** integers only
* **Defect / MDL:** floats allowed
* **Comparisons:** exact unless declared otherwise

Explicitly ban:

* epsilon-fuzzy carrier equality
* “close enough” ternary logic

---

## 6. Versioning & Semver Semantics

### Question

What constitutes a breaking change?

### Recommendation

Define now:

| Change             | Version Impact |
| ------------------ | -------------- |
| IO shape change    | MAJOR          |
| Invariant change   | MAJOR          |
| New op added       | MINOR          |
| Performance change | PATCH          |

And add:

> “GPU acceleration never changes the major version.”

---

## 7. Documentation as Spec, Not Narrative

### Decision

Docs describe **what must be true**, not *why it’s cool*.

### Recommendation

* Use Definition / Proposition / Invariant style
* Avoid metaphors in core docs
* Put intuition in separate notes

This prevents philosophical drift.

---

## 8. Dependency Freeze

### Question

What libraries are allowed?

### Recommendation

* Python stdlib
* NumPy
* pytest

Nothing else until Sprint 1 exit.

This prevents accidental coupling to ML frameworks.

---

## 9. Namespace Discipline

### Question

Can downstream projects extend core classes?

### Recommendation

* Core classes are **final** (conceptually)
* Extension via composition, not inheritance
* No monkey-patching

Document this explicitly.

---

## 10. What Happens When You’re Wrong

This is subtle but important.

### Question

How do you correct a foundational mistake?

### Recommendation

Define a **formal deprecation process** now:

1. Mark old behavior as deprecated
2. Add violation test explaining why
3. Introduce replacement
4. Remove in next major version

This keeps the theory coherent over time.

---

## 11. Minimal “Hello World” Canon

Decide what the **smallest complete example** is.

Recommendation:

```
Carrier → Kernel → Defect → MDL
```

If someone can’t understand that path, the core is too big.

---

## 12. Governance (Even If It’s Just You)

### Question

Who can change dashiCORE?

### Recommendation

Write this sentence:

> “Changes to dashiCORE require explicit justification against existing tests and invariants.”

Even solo projects benefit from this psychological barrier.

---

## 13. The One Irreversible Decision

This is the deepest one.

### Question

Is dashiCORE allowed to grow indefinitely?

### Recommendation (Strong)

No.

Define now:

> “dashiCORE is intentionally small.
> If in doubt, it does not belong here.”

This keeps it a **kernel**, not a framework.

---

## Final Pre-Start Checklist (Print This)

Before first commit, confirm:

* [ ] Normative vs illustrative clearly marked
* [ ] Equality vs equivalence formalised
* [ ] Fail-fast policy adopted
* [ ] Determinism guaranteed
* [ ] Float policy defined
* [ ] Versioning rules written
* [ ] Dependencies frozen
* [ ] Extension rules defined
* [ ] Deprecation path defined
* [ ] Core size intentionally capped

If all boxes are checked, you’re ready.

---

### Closing Thought (Worth Putting in the Repo)

> **Once shape and invariants are fixed, speed is easy.
> Once speed is fixed, shape is impossible to change.**
