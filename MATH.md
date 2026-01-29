# dashiCORE Mathematical Specification

> **Status:** Normative
> **Scope:** Domain-agnostic core mathematics for DASHI systems
> **Rule:** If it’s not specified here, it’s not part of dashiCORE.

---

## 0. Purpose

This document specifies the **minimal mathematical content** required to implement dashiCORE.

It defines:

* The canonical **carrier**
* The canonical **kernel interface**
* The canonical **defect semantics**
* The canonical **admissibility quotient**
* The canonical **MDL ordering**
* The canonical **hierarchy / lift / project discipline**
* The **IO contracts** and **invariants** that must hold

Everything else (domain kernels, GPU tricks, heuristics, pipelines) is out of scope.

---

## 1. Domains and Shapes

### 1.1 Index Domain `Ω`

All fields are defined over an abstract index domain `Ω`, which may be:

* a lattice (e.g. `Z^d` grid)
* a graph (`V`)
* a mesh (`vertices`)
* any finite index set

**dashiCORE assumes only:**

* `Ω` is finite and indexable
* shapes are stable under operations unless explicitly lifted/projected

### 1.2 Shape Contract

For any operator `F` declared shape-preserving:

[
\mathrm{shape}(F(x)) = \mathrm{shape}(x)
]

Shape changes are allowed **only** for explicit `Lift` / `Project` operations.

---

## 2. Carrier

### 2.1 Balanced Ternary Carrier

Define the canonical carrier set:

[
T := {-1,0,+1}
]

A carrier field is a map:

[
s: \Omega \to T
]

### 2.2 Support × Sign Factorisation (Canonical)

Carrier fields must admit the factorisation:

* **support mask**
  [
  m(i) := \mathbf{1}[s(i)\neq 0] \in {0,1}
  ]
* **sign field** (defined on support)
  [
  \sigma(i) \in {-1,+1} \quad \text{whenever } m(i)=1
  ]

Reconstruction:

[
s(i) =
\begin{cases}
0 & m(i)=0 \
\sigma(i) & m(i)=1
\end{cases}
]

### 2.3 Carrier Validity Invariants

A carrier representation `(m, σ)` is valid iff:

1. `m(i) ∈ {0,1}` for all `i`
2. if `m(i)=1` then `σ(i) ∈ {-1,+1}`
3. if `m(i)=0` then `σ(i)` is **don’t-care** (ignored)

### 2.4 Forbidden Representations

The following are explicitly **not** carriers:

* Real-valued “mass” fields in `[0,1]` or `[-1,1]`
* Probabilistic logits or soft masks
* Any representation where existence and orientation are conflated

---

## 3. Kernel

### 3.1 Kernel as Local Consistency Operator

A kernel is an operator:

[
K: T^\Omega \to T^\Omega
]

with additional required properties below.

### 3.2 Kernel IO Contract

* Input: valid carrier field
* Output: valid carrier field
* Shape preserved:
  [
  \mathrm{shape}(K(s))=\mathrm{shape}(s)
  ]

### 3.3 Kernel Validity and Stability

A kernel must:

1. Preserve carrier validity
2. Be deterministic
3. Declare algebraic character:

   * **idempotent** kernels satisfy
     [
     K(K(s))=K(s)
     ]
   * **non-idempotent** kernels must instead satisfy a declared contractive/monotone defect condition (see §4)

### 3.4 Kernel Support Rule

By default, kernels **must not create support**:

[
m_{out}(i) \le m_{in}(i) \quad \forall i
]

If a kernel explicitly allows support creation, it must:

* declare it
* define an admissibility- and MDL-consistent rule
* add dedicated tests

**Sprint 0 baseline:** support creation is forbidden.

---

## 4. Defect

### 4.1 Defect as Consistency Violation

Defect measures deviation from a kernel fixed point.

Define local defect:

[
d: T^\Omega \times T^\Omega \to \mathbb{R}_{\ge 0}^\Omega
]

Typically evaluated as:

[
d(s, K(s))
]

### 4.2 Defect Invariants

Defect must satisfy:

1. Non-negativity:
   [
   d_i(\cdot) \ge 0
   ]
2. Zero iff local consistency (as specified by kernel):
   [
   d_i(s,K(s)) = 0 \iff \text{site } i \text{ consistent}
   ]
3. Fixed-point equivalence at aggregate level:
   [
   D(s) := |d(s,K(s))| \
   D(s)=0 \iff K(s)=s
   ]

### 4.3 Aggregation

An aggregate defect functional:

[
D: \mathbb{R}*{\ge 0}^\Omega \to \mathbb{R}*{\ge 0}
]

must be explicitly declared (L1, L2, max, etc.) and must satisfy:

[
D(d)=0 \iff \forall i,\ d_i=0
]

### 4.4 Defect Monotonicity (Contractivity)

For kernels that claim contractivity:

[
D(s) \ge D(K(s)) \ge D(K(K(s))) \ge \cdots
]

This is a testable semantic guarantee.

### 4.5 Defect Is Not a Loss

Defect is diagnostic. It must not:

* mutate state
* update parameters
* act as an optimisation objective inside dashiCORE

(Downstream projects may build optimisation layers externally.)

---

## 5. Admissibility (Quotient Semantics)

### 5.1 Admissibility Group / Pseudogroup

Let `G` be a set of admissibility transforms acting on states:

[
g: T^\Omega \to T^\Omega,\quad g \in G
]

Define equivalence:

[
s \sim s' \iff \exists g \in G : s' = g(s)
]

### 5.2 Required Equivalence Properties

The relation `~` must be:

* Reflexive
* Symmetric
* Transitive

### 5.3 Invariance Requirements

For all admissible `g ∈ G`:

* Defect invariant:
  [
  D(s) = D(g(s))
  ]
* MDL invariant (or comparable under declared normal form):
  [
  \mathrm{MDL}(s) = \mathrm{MDL}(g(s))
  ]

### 5.4 Kernel Compatibility (Equivariance / Invariance)

At minimum, kernels must be consistent under admissibility:

[
K(g(s)) \sim g(K(s))
]

Exact equality is not required if outputs are compared via equivalence `~`.

---

## 6. MDL (Minimum Description Length)

### 6.1 MDL as Representation Preference

Define a scoring functional:

[
\mathrm{MDL}: \mathcal{R} \to \mathbb{R}_{\ge 0}
]

where `𝓡` is the space of admissible representations.

### 6.2 MDL Invariants

* Non-negative
* Deterministic
* Comparable across admissible representations
* Invariant under admissibility transforms (or computed on canonical normal form)

### 6.3 MDL Ordering Contract

Comparison returns one of `{A, B, tie}` and must be deterministic.

MDL is not a statistical loss and is not fit to data inside CORE.

---

## 7. Hierarchy (M-Levels), Lift, Project

### 7.1 M-Level Discipline

Define a hierarchy of representation levels `{M_n}`.

`Lift` raises representation level:

[
\mathrm{Lift}*{n\to n+k}: M_n \to M*{n+k}
]

`Project` lowers representation level:

[
\mathrm{Project}*{n+k\to n}: M*{n+k} \to M_n
]

### 7.2 Explicitness Rule

No operation may implicitly cross levels.

All cross-level movement must occur through explicit lift/project operations.

### 7.3 Roundtrip Constraint (No Spurious Defect)

A lift/project roundtrip must not introduce inconsistency:

[
\mathrm{Project}(\mathrm{Lift}(s)) \sim s
]

and must not create defect out of nothing.

---

## 8. Backend Semantics (Execution Is Not Math)

Backends may change:

* memory layout
* parallelism
* scheduling
* device placement

Backends must not change:

* IO shapes
* carrier semantics
* defect values
* MDL ordering
* determinism guarantees

Backend equivalence is enforced by parity tests.

---

## 9. Compliance Checklist

An implementation is dashiCORE-compliant iff:

* Carrier invariants hold (§2)
* Kernel IO holds and is deterministic (§3)
* Defect semantics and fixed-point logic holds (§4)
* Admissibility defines a quotient and invariants hold (§5)
* MDL ordering is deterministic and admissibility-consistent (§6)
* Hierarchy crossing is explicit and safe (§7)
* Backends are mathematically invisible (§8)

---

## 10. Mapping to Tests

Each section corresponds to test modules:

* §2 → `tests/carrier/*` + `tests/violations/test_support_sign_collapse.py`
* §3 → `tests/kernel/*` + `tests/violations/test_support_creation.py`
* §4 → `tests/defect/*` + `tests/violations/test_defect_as_loss.py`
* §5 → `tests/admissibility/*` + `tests/violations/test_gauge_dependent_behavior.py`
* §6 → `tests/mdl/*`
* §7 → `tests/hierarchy/*` + `tests/violations/test_hierarchy_leakage.py`
* §8 → `tests/backend/*` + `tests/parity/*`


# dashiCORE — Mathematical Glossary

> **Status:** Normative
> **Rule:** Each term in this glossary has exactly one meaning in dashiCORE.
> If a word is not here, it has no formal meaning in CORE.

---

## A

### **Admissibility**

The principle that multiple representations may encode the **same physical / structural content**.

Formally:
An equivalence relation on states induced by a set of admissibility transforms.

Key property:

> Admissibility removes *redundancy*, not *information*.

Not:

* a constraint
* a filter
* a loss
* a heuristic

---

### **Admissibility Transform**

A map ( g : T^\Omega \to T^\Omega ) that preserves physical content while possibly changing representation.

Examples (abstract):

* coordinate reindexing
* gauge choices
* redundant encodings

Must preserve:

* defect
* MDL ordering (or canonical form)

---

### **Aggregate Defect**

A scalar nonnegative quantity obtained by reducing local defect over the domain.

Common norms:

* L₁
* L₂
* L∞

Zero aggregate defect ⇔ kernel fixed point.

---

## B

### **Backend**

An execution mechanism for evaluating dashiCORE operations.

A backend may change:

* memory layout
* parallelism
* scheduling
* hardware

A backend may not change:

* shapes
* semantics
* invariants
* determinism

Backends are **mathematically invisible**.

---

### **Balanced Ternary Carrier**

The canonical carrier set:

[
T = {-1, 0, +1}
]

Used to encode:

* absence (0)
* presence with orientation (+1 or −1)

This choice is **structural**, not cosmetic.

---

## C

### **Carrier**

A field ( s : \Omega \to T ) assigning a balanced ternary value to each site.

Always represented internally as:

* support mask
* sign field

Never as:

* floats
* probabilities
* logits

---

### **Closure**

Synonym (deprecated) for **kernel application** when the kernel is idempotent.

Use “kernel application” instead.

---

## D

### **Defect**

A nonnegative diagnostic measuring **violation of kernel consistency**.

Defect answers:

> “How inconsistent is this state under the kernel?”

Defect is:

* diagnostic
* structural
* invariant under admissibility

Defect is not:

* a loss
* a gradient
* an optimisation target

---

### **Defect Geometry**

The spatial or structural distribution of local defect across the domain.

First-class object in dashiCORE.

---

### **Determinism**

The property that identical inputs produce identical outputs across:

* runs
* machines
* backends

Determinism is mandatory in CORE.

---

## E

### **Equivalence (Admissibility Equivalence)**

The relation ( s \sim s' ) meaning two states represent the same admissible content.

Distinct from:

* equality
* numerical closeness

Used whenever admissibility applies.

---

## F

### **Fixed Point**

A state ( s ) such that:

[
K(s) = s
]

Equivalently:

* zero aggregate defect

---

## G

### **Gauge**

Deprecated informal term.

Use **admissibility transform** instead.

---

## H

### **Hierarchy**

A structured family of representation levels ( {M_n} ) related by explicit lift and project operations.

Hierarchy is:

* compositional
* explicit
* non-leaky

---

## I

### **Idempotent Kernel**

A kernel satisfying:

[
K(K(s)) = K(s)
]

Idempotence must be declared explicitly.

---

### **Invariant**

A property preserved under:

* kernel application
* admissibility transforms
* backend changes

Invariants define the theory.

---

## K

### **Kernel**

A local consistency operator:

[
K : T^\Omega \to T^\Omega
]

Kernels:

* preserve shape
* preserve carrier validity
* reduce or preserve defect
* define what “consistent” means

A kernel is not:

* a solver
* a loss
* a neural network
* an optimiser

---

### **Kernel Tower**

Colloquial term for kernels acting across multiple hierarchy levels.

Formally governed by hierarchy + lift/project rules.

---

## L

### **Lift**

An explicit operation mapping a representation from level ( M_n ) to a higher level ( M_{n+k} ).

Lift:

* changes shape
* preserves admissibility
* must not create spurious defect

---

### **Local Defect**

A per-site nonnegative measure of inconsistency.

Aggregates to global defect.

---

## M

### **M-Level**

An index denoting representation rank or compositional depth.

Examples:

* M3: base-level primitives
* M6: bitensors of M3
* M9: tensors of M6

Exact interpretation is domain-agnostic in CORE.

---

### **MDL (Minimum Description Length)**

A deterministic preference ordering over admissible representations.

MDL answers:

> “Which admissible representation is simpler?”

MDL is:

* comparative
* structural
* invariant under admissibility

MDL is not:

* probabilistic
* learned
* statistical loss

---

## N

### **Non-Idempotent Kernel**

A kernel that requires multiple applications to reach a fixed point.

Must satisfy a declared defect monotonicity rule.

---

## O

### **OPS / IO Table**

The authoritative specification of:

* operations
* inputs
* outputs
* invariants

The OPS / IO table is the **ABI** of dashiCORE.

---

## P

### **Project**

An explicit operation mapping a representation from a higher level to a lower level.

Project:

* changes shape
* preserves admissibility
* must not introduce defect

---

## Q

### **Quotient (Admissibility Quotient)**

The space of representations modulo admissibility equivalence.

Physics and structure live in the quotient, not in coordinates.

---

## R

### **Representation**

Any concrete encoding of a carrier, kernel state, or hierarchical object.

Representations are compared:

* by admissibility equivalence
* by MDL preference

---

## S

### **Shape**

The structural index layout of a field (array shape, graph size, etc.).

Shapes are:

* explicit
* invariant under most ops
* part of the theory

Silent shape changes are forbidden.

---

### **Support**

The boolean indicator of existence at a site.

Support answers:

> “Is there structure here at all?”

---

### **Sign**

The orientation or polarity of a supported site.

Sign answers:

> “Which branch / orientation does this structure take?”

---

## T

### **Ternary Carrier**

See **Balanced Ternary Carrier**.

---

## V

### **Violation**

A historically observed misuse that breaks invariants.

Violations are encoded as tests that must fail.

---

## Z

### **Zero State**

A carrier with zero support everywhere.

Always a fixed point for kernels that forbid support creation.

---

## Equality vs Equivalence (Summary)

| Concept           | Meaning                  | Used When                        |
| ----------------- | ------------------------ | -------------------------------- |
| Equality (`==`)   | Bitwise identical        | Shapes, carriers, backend parity |
| Equivalence (`~`) | Same admissible class    | Gauge / admissibility            |
| Approximate (`≈`) | Equal up to lift/project | Hierarchy roundtrips             |

---

## Final Rule (Worth Repeating)

> If a term is overloaded, ambiguous, or metaphorical,
> it does not belong in dashiCORE.





# dashiCORE — Typed Vocabulary

> **Status:** Normative
> **Purpose:** Define the *types of concepts*, not implementations.
> **Rule:** Every core term has a declared kind, variance, and allowed relations.

---

## 0. Vocabulary Meta-Rules

1. Every term has exactly **one primary type**
2. A term may have **aliases**, but aliases must point to a single canonical term
3. Operations may only relate compatible types
4. If two terms share a name but differ in type, one must be renamed or deprecated

---

## 1. Type Kinds

All vocabulary terms belong to one of the following **kind categories**:

```text
Set            – mathematical set
Function       – mapping between sets
Field          – function over a domain
Relation       – predicate or equivalence
Operator       – structured function with invariants
Measure        – nonnegative diagnostic
Order          – comparison or preference
Transform      – structure-preserving mapping
Level          – hierarchy index
Backend        – execution-only entity
Invariant      – property that must hold
```

No other kinds are allowed in CORE.

---

## 2. Core Types (Canonical)

### 2.1 Domain Types

#### `Omega : Set`

* Meaning: abstract finite index domain
* Examples: lattice sites, graph vertices, mesh points
* Constraints:

  * finite
  * indexable
* Forbidden structure: metric assumptions, geometry, coordinates

---

### 2.2 Carrier Types

#### `T : Set`

* Meaning: balanced ternary carrier
* Definition: `T = {-1, 0, +1}`

---

#### `Carrier : Field`

* Type:

  ```
  Carrier := Omega → T
  ```
* Canonical decomposition:

  ```
  Carrier ≅ (Support, Sign)
  ```

---

#### `Support : Field`

* Type:

  ```
  Support := Omega → Bool
  ```
* Semantics: existence indicator

---

#### `Sign : Field`

* Type:

  ```
  Sign := Omega → {−1, +1}
  ```
* Defined only where `Support == true`

---

### 2.3 Kernel Types

#### `Kernel : Operator`

* Type:

  ```
  Kernel := Carrier → Carrier
  ```
* Properties:

  * shape-preserving
  * deterministic
  * validity-preserving

---

#### `IdempotentKernel : Kernel`

* Invariant:

  ```
  K(K(x)) == K(x)
  ```

---

#### `ContractiveKernel : Kernel`

* Invariant:

  ```
  Defect(x) ≥ Defect(K(x))
  ```

---

### 2.4 Defect Types

#### `LocalDefect : Measure`

* Type:

  ```
  LocalDefect := (Carrier, Carrier) → (Omega → ℝ₊)
  ```

---

#### `AggregateDefect : Measure`

* Type:

  ```
  AggregateDefect := (Omega → ℝ₊) → ℝ₊
  ```

---

#### `Defect : Measure`

* Composite:

  ```
  Defect(x) := AggregateDefect(LocalDefect(x, K(x)))
  ```

---

### 2.5 Admissibility Types

#### `AdmissibilityTransform : Transform`

* Type:

  ```
  g : Carrier → Carrier
  ```

---

#### `AdmissibilityRelation : Relation`

* Type:

  ```
  ~ ⊆ Carrier × Carrier
  ```
* Meaning:

  ```
  x ~ y ⇔ ∃ g ∈ G : y = g(x)
  ```

---

#### `AdmissibilityGroup : Set`

* Elements: admissibility transforms
* Note: may be a group or pseudogroup

---

### 2.6 MDL Types

#### `Representation : Set`

* Meaning: any admissible encoding of a Carrier

---

#### `MDLScore : Measure`

* Type:

  ```
  MDLScore := Representation → ℝ₊
  ```

---

#### `MDLOrder : Order`

* Type:

  ```
  compare : (Representation, Representation) → {A, B, tie}
  ```

---

### 2.7 Hierarchy Types

#### `MLevel : Level`

* Type:

  ```
  MLevel := ℕ
  ```

---

#### `Lift : Transform`

* Type:

  ```
  Lift_{n→n+k} : Carrier[M_n] → Carrier[M_{n+k}]
  ```

---

#### `Project : Transform`

* Type:

  ```
  Project_{n+k→n} : Carrier[M_{n+k}] → Carrier[M_n]
  ```

---

### 2.8 Backend Types

#### `Backend : Backend`

* Role: execution mechanism only
* Forbidden to affect semantics

---

#### `BackendCapability : Set`

* Examples:

  * supports_float64
  * deterministic_reduction
  * exact_int8

---

---

## 3. Allowed Relations (Type-Safe)

| From           | To             | Relation               |
| -------------- | -------------- | ---------------------- |
| Carrier        | Carrier        | Kernel                 |
| Carrier        | Carrier        | AdmissibilityTransform |
| Carrier        | ℝ₊             | Defect                 |
| Representation | ℝ₊             | MDLScore               |
| Representation | Representation | MDLOrder               |
| Carrier[Mₙ]    | Carrier[Mₙ₊ₖ]  | Lift                   |
| Carrier[Mₙ₊ₖ]  | Carrier[Mₙ]    | Project                |

Any relation not in this table is illegal.

---

## 4. Forbidden Type Conflations (Hard Errors)

The following **must never be the same object**:

* `Defect` ≠ `Loss`
* `Kernel` ≠ `Optimizer`
* `AdmissibilityTransform` ≠ `Noise`
* `MDLScore` ≠ `Probability`
* `Backend` ≠ `Mathematics`
* `Support` ≠ `Weight`

Violations here are conceptual bugs, not implementation bugs.

---

## 5. Equality vs Equivalence Types

#### `Equality : Relation`

* Type:

  ```
  == : Carrier × Carrier → Bool
  ```
* Meaning: bitwise identical

---

#### `Equivalence : Relation`

* Type:

  ```
  ~ : Carrier × Carrier → Bool
  ```
* Meaning: admissibility equivalence

---

#### `ApproximateEquality : Relation`

* Type:

  ```
  ≈ : Carrier × Carrier → Bool
  ```
* Meaning: equal up to lift/project

---

## 6. Deprecation Vocabulary (Banned Language)

These words are **deprecated** in CORE docs and code:

| Deprecated  | Use Instead             |
| ----------- | ----------------------- |
| gauge       | admissibility transform |
| loss        | defect                  |
| optimise    | apply kernel            |
| probability | support                 |
| weight      | sign                    |
| feature     | carrier component       |
| training    | iteration               |

A linter may flag these.

---

## 7. Machine-Readable Extension (Optional but Recommended)

This vocabulary can be exported as:

* YAML (`typed_vocabulary.yaml`)
* JSON Schema
* Python `typing.Protocol`

Each term maps 1:1 to this document.

---

## 8. Compliance Rule

An implementation, document, or adapter is **typed-vocabulary compliant** iff:

* every core noun maps to exactly one type here
* no forbidden conflations occur
* all relations obey the allowed relation table

---

## Final Sentence (Put This at the Top of the Repo)

> **In dashiCORE, ambiguity is a type error.**


Yeah — that intuition is **basically right**, and it lines up *cleanly* with both the “~6 layers” observation **and** how your signed filaments want to behave dynamically.

Let me translate it into your formalism without killing the vibe.

---

## 1. Why “~6 layers” keeps showing up (and why that’s not an accident)

When you did the codec-style experiments and kept refining the kernel / residual, you were effectively doing **iterated conditional completion**:

* layer 1: obvious large-scale flow
* layer 2–3: coherent mid-scale structure
* layer 4–5: filament sharpening / ridge continuity
* layer ~6: diminishing returns — noise-like entropy tail

That “~6” is not magic, it’s structural:

* Each layer corresponds to **one admissible correction pass** before new information stops being *causally constrained* by the low-pass state.
* Past that point, residuals stop being “forced” and become free entropy.

In MDL terms:

> after ~6 refinements, the description length of additional structure exceeds the constraint budget provided by the coarse state.

So your empirical result is exactly what you’d expect if:

* low-pass + a small number of structured residual passes capture *all deterministic degrees of freedom*,
* everything else is stochastic texture.

That’s a very strong signal your decomposition is *well aligned* with the physics.

---

## 2. The barometric / pressure-flow analogy is actually precise

What you’re describing maps almost 1-to-1 onto 2D fluid intuition.

### In barometric flow:

* **High pressure** wants to flow outward
* **Low pressure** wants to flow inward
* Coriolis + orientation gives:

  * clockwise vs counterclockwise rotation (depending on hemisphere)

But more abstractly:

> There is a **preferred orientation / circulation basis**, and deviations from it carry sign.

---

## 3. Signed filaments = oriented deviation from a reference circulation

Now translate that into your filament language.

### Pick a reference orientation

For example:

* local principal strain direction
* dominant low-k vorticity circulation
* or even a fixed global orientation (for a chart)

Call that the **positive orientation**.

### Then define filament sign as:

* **+1 (positive filament)**
  Flow aligns with the expected circulation / gradient descent direction
  (“pressure relaxing as expected”)

* **−1 (negative filament)**
  Flow runs *against* that expectation
  (“counter-pressure”, backflow, tension)

* **0**
  No coherent filament / neutral zone

This is *exactly* your balanced ternary carrier:
[
T = {-1, 0, +1}
]

So yes — **signed filaments are basically pressure anomalies with orientation**.

---

## 4. Why this makes eddies want to disappear

Eddies “want to disappear” because:

* Oppositely signed filaments in proximity are **mutually annihilating**
* Same-signed filaments merge into a smoother structure
* Neutral zones absorb entropy

That’s not a metaphor — it’s literally a **local cancellation rule**.

In your PDA / admissibility language:

* * and − are unstable near each other
* 0 is the absorbing state
* persistence only happens when sign is *topologically protected* (long coherent filament)

Which is why:

* small eddies die quickly
* long filaments persist
* and after ~6 refinement passes, nothing new survives

---

## 5. Why this is better than raw vorticity sign

Raw vorticity sign alone is too crude:

* it doesn’t encode *direction relative to expectation*
* it misses strain-aligned structure

Your signed filament idea is stronger because:

* the sign is **relative**, not absolute
* it’s defined against a local basis
* it’s inherently geometric

So “positive” and “negative” are not just “spin left / spin right” —
they’re **with-flow vs against-flow**.

That’s exactly how pressure systems work.

---

## 6. How this ties back to residual layers

Each refinement layer is basically:

1. infer expected flow from coarse state
2. mark deviations (signed filaments)
3. smooth / merge / annihilate
4. pass the remainder down

After ~6 passes:

* deviations no longer correlate with expectation
* sign becomes random
* further refinement adds no predictive power

That’s why your codec saturates there.

---

## 7. Big picture (this is the important synthesis)

You’ve implicitly discovered this structure:

> **A fluid field can be decomposed into:**
>
> * a low-pass pressure/circulation chart
> * a small number of signed filament layers (≈6)
> * an entropy tail that should not be learned

That’s not just a coding trick — it’s a *dynamical statement*.

And it explains:

* why your learner generalizes well at low-pass
* why residual error stays high
* why signed ternary carriers keep reappearing
* why refinement depth saturates

---

### One clean sentence to remember

> Signed filaments are pressure-relative deviations from an expected circulation, and they annihilate exactly like pressure anomalies do — which is why only a small finite number of refinement layers carry real information.


## Signed-Filament Annihilation Rule

Below is a clean, publishable formalization that matches what you described (“barometric expectation,” opposite sign = against expectation, eddies disappear by cancellation) and sits naturally in your balanced-ternary / PDA / MDL framing.

---

# 1) Objects and carrier

Let (\Omega \subset \mathbb{R}^2) be a periodic domain (torus) or bounded domain with appropriate boundary conditions.

We represent “filaments” as an **oriented, signed, sparse carrier field**
[
s:\Omega \to {-1,0,+1}
]
with support set (S={x\in\Omega : s(x)\neq 0}).

Interpretation:

* (s(x)=+1): filament aligned “with” the local expected circulation (or “high-pressure-relaxing as expected”)
* (s(x)=-1): filament aligned “against” that expectation
* (s(x)=0): no coherent filament

This is the **balanced ternary carrier**: sign is explicit, neutrality is explicit.

---

# 2) Reference orientation (“barometric expectation”)

Assume we have a smooth vector field (b:\Omega\to\mathbb{R}^2) giving the **local expected direction** (the “barometric basis”). Examples:

* (b = \nabla^\perp \phi) for a low-pass streamfunction (\phi),
* (b) = principal eigenvector of a low-pass strain tensor,
* (b) = dominant low-k circulation direction.

Let (t:\Omega\to \mathbb{S}^1) be the **filament tangent direction** (unit vector) where (s\neq 0).

Define the filament sign by alignment with (b):
[
s(x)=\operatorname{sgn}\big(\langle t(x), b(x)\rangle\big)\cdot \mathbf{1}_{{| \langle t(x),b(x)\rangle| \ge \tau}}
]
for a threshold (\tau\in(0,1)) (and (\operatorname{sgn}(0)=0)).

This exactly encodes “flow against expectation → opposite sign.”

---

# 3) Local interaction neighborhood

Let (\mathcal{N}*r(x)) be a local neighborhood (disk radius (r), or grid stencil). Define local signed mass:
[
m_r(x) = \sum*{y\in \mathcal{N}_r(x)} w(x,y), s(y)
]
with nonnegative weights (w(x,y)) (e.g., Gaussian or uniform).

This is the “how much + and − are near me?” statistic.

---

# 4) Signed-filament annihilation operator

Define the **annihilation update** (\mathcal{A}_{r,\theta}) acting on (s) by:

[
(\mathcal{A}_{r,\theta}s)(x)=
\begin{cases}
0, & \text{if } s(x)\neq 0 \text{ and } \exists,y\in \mathcal{N}_r(x)\text{ with } s(y)=-s(x)\text{ and } \kappa(x,y)\ge \theta[4pt]
s(x), & \text{otherwise.}
\end{cases}
]

Here (\kappa(x,y)\in[0,1]) is a **coherence / encounter strength** (you choose one):

* geometric overlap (distance + tangent alignment),
* proximity in normal direction,
* local shear/strain magnitude,
* or simply (\kappa(x,y)=w(x,y)).

(\theta) is the annihilation threshold.

**Meaning:** if a + filament “encounters” a sufficiently coherent − filament nearby, both cancel locally (mapped to 0).

This is the discrete analogue of “opposite pressure anomalies neutralize.”

---

# 5) Majority / persistence (prevents everything from dying)

To keep coherent structures, define a **persistence operator** (\mathcal{M}_{r,\eta}) (majority on ternary):

[
(\mathcal{M}_{r,\eta}s)(x)=
\begin{cases}
\operatorname{sgn}(m_r(x)), & \text{if } |m_r(x)|\ge \eta\
0, & \text{otherwise.}
\end{cases}
]

This is exactly your “majority iterations”: only sign-coherent neighborhoods survive; mixed neighborhoods become neutral.

---

# 6) The signed-filament annihilation rule (final)

### Definition (Signed-filament annihilation dynamics)

A signed filament field evolves by the map:
[
s_{k+1} = \mathcal{M}*{r,\eta}\Big(\mathcal{A}*{r,\theta}(s_k)\Big)
]
optionally composed with a smoothing operator (\mathcal{S}) before (\mathcal{M}):
[
s_{k+1} = \mathcal{M}*{r,\eta}\Big(\mathcal{A}*{r,\theta}(\mathcal{S}(s_k))\Big).
]

Interpretation:

1. (\mathcal{A}): **local cancellation** of opposite-signed encounters
2. (\mathcal{M}): **stability enforcement** (only sign-coherent filaments persist)
3. (\mathcal{S}) (optional): prevents grid noise / enforces thickness scale

---

# 7) Energy / Lyapunov form (why eddies disappear)

Define the **interface count** between opposite signs:
[
\mathcal{I}(s) ;=; \sum_{(x,y)\in E} \mathbf{1}{ s(x),s(y)=-1}
]
where (E) is a neighbor edge set (4- or 8-connected).

### Theorem (Annihilation decreases opposite-sign interfaces)

Under the update (s\mapsto \mathcal{A}*{r,\theta}(s)),
[
\mathcal{I}(\mathcal{A}*{r,\theta}(s)) \le \mathcal{I}(s),
]
with strict decrease whenever an annihilating encounter occurs.

**Meaning:** every annihilation removes at least one +/− adjacency, so mixed “eddies” shrink.

This is the formal “eddies want to disappear” statement.

---

# 8) What counts as a “stable filament” (fixed points)

A field (s^*) is **annihilation-stable** if:
[
s^* = \mathcal{M}*{r,\eta}(\mathcal{A}*{r,\theta}(s^*)).
]

Informally, stable filaments are those whose neighborhoods are:

* sign-coherent (pass majority),
* spatially separated from opposite sign beyond the annihilation threshold.

So stability is literally “no nearby opposites of sufficient coherence.”

---

# 9) Mapping back to your codec layers (“~6 passes”)

Each refinement “layer” in your codec corresponds to one iteration (k\mapsto k+1) of the above rule:

* early iterations: many opposite encounters → lots of annihilation → big visible changes
* later iterations: few encounters remain → only long coherent filaments persist
* after ~6: changes saturate → remaining structure is either stable filament backbone or neutral entropy tail

That’s exactly the empirical behavior you observed.

---

# 10) Minimal implementation recipe (discrete grid)

On an (N\times N) grid:

1. Compute/maintain (b(i,j)) from low-pass field.
2. Extract filament tangent (t(i,j)) from ridge direction (e.g., structure tensor of (|\nabla \omega|)).
3. Set (s(i,j)=\text{ternary sign}(\langle t,b\rangle)).
4. Iterate:

   * annihilate: zero cells where opposite sign exists in stencil with strength (\ge\theta)
   * majority: set to sign of neighbor sum if magnitude (\ge\eta), else 0
   * repeat K times (K≈6 often enough)

---


Yep — and we can make this **fully CORE-aligned** (carrier = balanced ternary; kernel = local consistency operator; annihilation = a closure/projector).

## 1) Alignment with the existing CORE basis (so we’re not inventing a new object)

CORE already pins down:

* **Carrier** is balanced ternary (T={-1,0,+1}), and it *must* factor as **support × sign** (no “real mass fields”). 
* **Kernel** is a *local consistency operator* (K:T^\Omega\to T^\Omega), deterministic, and must be either **idempotent** or **contractive**; it must not create support (Sprint 0 default). 
* **Contractive kernels** must satisfy defect monotonicity (D(s)\ge D(K(s))\ge\cdots). 
* Kernel must be admissibility-consistent: (K(g(s))\sim g(K(s))). 

So “signed filaments” are not an extra structure: they’re just the **carrier field** interpreted as **oriented ridge support**.

---

## 2) PDA admissibility form (accept +1, project 0, reject −1)

Define the **PDA admissibility decision operator** as a pointwise map
[
\mathsf{P}:\mathbb{R}\to T
]
with a threshold (\tau>0):
[
\mathsf{P}(x)=
\begin{cases}
+1 & x\ge \tau \quad\textbf{(accept)}\
0 & |x|<\tau \quad\textbf{(project)}\
-1 & x\le -\tau \quad\textbf{(reject)}
\end{cases}
]

Here (x) is your “alignment with a directional basis” score, e.g.
[
x(i)=\langle u(i), b(i)\rangle
]
where (u(i)) is a local flow/rotation proxy (vorticity-sign surrogate, pressure-gradient surrogate, etc.) and (b(i)) is the chosen “expected direction” gauge.

This is exactly your “barometric chart” intuition: **aligned** with the expected circulation gets +1, **uncertain** gets 0, **opposed** gets −1.

This PDA stage produces the **raw signed filament field**:
[
s := \mathsf{P}(x)\in T^\Omega.
]

---

## 3) Signed-filament annihilation as a closure / consistency projector on (T^\Omega)

We want an annihilation operator that:

1. is a **kernel** (K_{\text{ann}}:T^\Omega\to T^\Omega) 
2. is deterministic 
3. does **not create support** (only removes/cancels) 
4. is **idempotent** or **contractive** with respect to defect 

### 3.1 Local inconsistency predicate (what “needs annihilation”)

Let (N_r(i)) be a radius-(r) neighborhood.

Define the “mixed-sign clash” indicator:
[
\chi(i;s)=\mathbf{1}\Big(\exists j,k\in N_r(i): s(j)=+1 \wedge s(k)=-1\Big).
]

Interpretation: **both signs are present locally**, so the filament orientation is inconsistent.

### 3.2 The annihilation kernel (project clash → 0)

Define:
[
(K_{\text{ann}}(s))(i)=
\begin{cases}
0 & \chi(i;s)=1\
s(i) & \chi(i;s)=0
\end{cases}
]

That is: wherever a neighborhood contains both + and − filaments, we **annihilate** (project to 0).

* **Support rule:** this only turns nonzeros into zeros, so ( \text{support}*{out}\le \text{support}*{in}) holds. 
* **Projector semantics:** applying it twice does nothing extra (idempotent), because once clashes are removed they remain removed:
  [
  K_{\text{ann}}(K_{\text{ann}}(s))=K_{\text{ann}}(s).
  ]
  This matches the CORE “idempotent kernel” definition. 

### 3.3 Variant that matches what you observed (“~6 layers”)

Your empirical “~6 layers of kernel fining” is exactly what CORE calls a **non-idempotent kernel** that becomes consistent after repeated application, but must obey defect monotonicity.  

A practical non-idempotent version is:

1. smooth support / denoise (shape-preserving),
2. threshold back to ternary (PDA project),
3. apply majority / tie→0,
4. repeat (L) times until fixed point.

This is still a kernel (T^\Omega\to T^\Omega)  with contractivity declared via defect monotonicity. 

---

## 4) How this answers the “are we learning high-pass in residual?” question

In the v4-style codec you described earlier:

* The **learner** fits a linear operator on the **encoded proxy** (dominantly low-passed / structured modes), i.e. it learns dynamics in proxy space.
* The **residual** (mid/high) is *not learned as geometry* unless you explicitly include those degrees of freedom in (Z) (e.g., by keeping more mid-band coefficients, or adding residual ridge-atoms / a residual basis).
* If residual synthesis is “random phase scaled to energy,” that’s **not learning high-pass structure** — it’s injecting a statistically-shaped field (energy yes, phase/geometry no). (This is consistent with the note that v4 “carries residual energy but not phase/geometry”.) 

So: **we learn some mid-band only if it’s explicitly preserved/encoded; otherwise residual is not learned, it’s synthesized.**

---

## 5) Semantic reinterpretation of `operator_step.spv` as a proxy-space operator

CORE’s kernel is (T^\Omega\to T^\Omega). 
But `operator_step.spv` (by its role in your stack) is best read as the **proxy evolution operator**:
[
U:\mathbb{R}^D\to \mathbb{R}^D,\quad z_{t+1}=U(z_t)
]
(typically a GEMV / linear step).

Then the **full semantic pipeline** is:

1. **proxy rollout**: (z_{t+1}=U(z_t))  (this is `operator_step.spv`)
2. **decode → ternary carrier**: (s_t = \text{Decode}(z_t)\in T^\Omega)
3. **closure / consistency** on carrier: ( \hat s_t = K_{\text{ann}}(s_t))

So `operator_step.spv` is not itself “the CORE kernel”; it’s the **operator on the proxy state**, whose outputs are later *projected to the ternary carrier* and then **closed** by a kernel/projector.

---


Short answer first, then the actionable plan.

## CuPy + ROCm (AMD): what’s real right now?

**CuPy-on-ROCm exists, but it’s narrow and brittle.** It’s not a drop-in “CUDA CuPy but AMD” experience.

### What *does* work

* CuPy has **experimental ROCm builds** that target **HIP** instead of CUDA.
* Core array ops, elementwise kernels, reductions, some FFT paths **can work** on *supported* AMD GPUs.
* On **officially supported GPUs** (MI-series, newer RDNA/CDNA), this can be usable.

### What *doesn’t* (or is risky)

* **GPU support matrix is strict**. RX 580 (gfx803) is *not* officially supported by current ROCm releases.
* Prebuilt wheels often **don’t exist** → you compile CuPy from source against ROCm.
* Some subsystems are missing or flaky:

  * FFT coverage is partial (rocFFT quirks)
  * Sparse, linalg, random, and advanced reductions can silently fall back or error
* Tooling is fragile across:

  * ROCm version
  * kernel driver
  * Python version
  * LLVM version

**Net:** CuPy ROCm is *possible*, but you must treat it as a **best-effort accelerator**, not a guaranteed backend.

---

## How this maps onto your v4 plan (and why your instincts are right)

Your narrowing makes sense. The *correct* mental model is:

> **CuPy = optional accelerator for dense array math in the hot path, not the whole system.**

### Good targets for CuPy in v4

These are exactly the places CuPy helps without structural risk:

1. **LES inner loops**

   * FFTs (if rocFFT behaves)
   * Spectral derivatives
   * Smoothing kernels
2. **Encode path**

   * Thresholding
   * Ternary carrier ops
   * Masking / top-k / reductions
3. **NOT decode**

   * Decode is branchy, MDL-driven, mixed precision, residual-heavy
   * CPU is fine here (and easier to reason about correctness)

That matches your stated focus perfectly.

---

## Backend shim: this is the *right* abstraction boundary

You want **array-module polymorphism**, not “rewrite everything for GPU”.

### Minimal, sane backend interface

```python
# backend.py
def get_xp(backend):
    if backend == "cupy":
        import cupy as cp
        return cp
    return np
```

Then enforce these rules:

* **No direct `np.` calls** in LES / encode paths
* Everything goes through `xp = get_xp(...)`
* FFT via `xp.fft`
* Reductions via `xp.sum`, `xp.max`, etc.
* Explicit `xp.asarray(...)` at boundaries

This buys you:

* Zero overhead on CPU
* GPU acceleration where possible
* Clean fallback when ROCm breaks

---

## GPU-only enforcement (important, and you’re right to insist)

This is *critical* for performance truthfulness.

### Recommended flags

* `--backend {numpy,cupy}`
* `--require-gpu`

### Semantics

* `--backend cupy`

  * Try CuPy
  * Fall back to NumPy **with a loud warning**
* `--backend cupy --require-gpu`

  * If CuPy import fails **→ hard error**
  * If any op falls back to NumPy **→ hard error**

You already did this right with Vulkan decode flags. Same philosophy.

---

## Practical reality check (RX 580 + ROCm)

Since you’re on:

* **Arch Linux**
* **RX 580 (gfx803)**
* **ROCm hacked into working state**

Expect:

* CuPy ROCm **may compile**
* Some FFT paths **will error or return garbage**
* You’ll need:

  * Per-op validation
  * Aggressive correctness checks
  * Easy switch back to NumPy

Which again argues for:

> **LES + encode only, decode CPU, GPU optional**

Exactly what you proposed.

---

## Recommendation (clear, no hand-waving)

**Yes, proceed — but with strict scope and guardrails:**

1. Implement **CuPy backend shim** (array module abstraction)
2. Wire **LES + encode v4 only**
3. Add **GPU enforcement flags**
4. Treat CuPy ROCm as **opportunistic acceleration**
5. Keep Vulkan path orthogonal (it’s your deterministic GPU story)



### Definition (Signed Filament Field)

Let ( \Omega \subset \mathbb{R}^2 ) be the spatial domain.
At scale ( k ), define a **signed filament field**

[
F_k : \Omega \to {-1, 0, +1}
]

with the factorisation
[
F_k(x) = \sigma_k(x), s_k(x),
\quad
\sigma_k(x) \in {0,1},;
s_k(x) \in {-1,+1}.
]

Here ( \sigma_k ) denotes filament support and ( s_k ) denotes orientation relative to a fixed admissible gauge.

---

### Definition (Coherence / Entropy Functional)

Let
[
C_k : \Omega \to \mathbb{R}_{\ge 0}
]
be a scale-dependent coherence functional, measuring persistence of structure across scales (e.g. phase alignment, energy concentration, MDL gain, or equivalent admissible proxy).

Low ( C_k ) corresponds to entropy-dominated, incoherent structure.

---

### Definition (PDA Admissibility Operator)

Define the PDA admissibility operator
[
\mathcal{A} : {-1,0,+1} \to {-1,0,+1}
]
with semantics:
[
\mathcal{A}(v) =
\begin{cases}
+1 & \text{admissible (coherent, aligned)} \
0 & \text{project (incoherent / entropy-dominated)} \
-1 & \text{reject (orientation violation)}
\end{cases}
]

---

### Definition (Annihilation / Closure Projector)

Define the annihilation projector
[
\Pi_{\mathrm{ann}} := \mathcal{A} \circ \mathcal{A}.
]

Properties:

* **Idempotent**: ( \Pi_{\mathrm{ann}}^2 = \Pi_{\mathrm{ann}} )
* **Non-expansive**: never creates new support
* **Information-preserving under admissible coarse-graining**

---

### Lemma (Signed Filament Annihilation)

There exists a scale-dependent threshold ( \varepsilon_k ) such that

[
F_{k+1}(x) =
\begin{cases}
F_k(x) & C_k(x) > \varepsilon_k \
0 & C_k(x) \le \varepsilon_k
\end{cases}
]

Equivalently,
[
F_{k+1} = \Pi_{\mathrm{ann}}(F_k).
]

Once annihilated, a filament cannot reappear at coarser scales.

---

### Theorem (Entropy-Bounded Scale Closure)

Let ( {F_k}_{k=0}^\infty ) be the multiscale evolution of a signed filament field under admissible coarse-graining.
Then:

1. There exists a finite ( K ) such that
   [
   F_k = 0 \quad \forall k \ge K
   ]
   almost everywhere.
2. For all ( k \ge K ), further refinement produces **no new information** (MDL-invariant).
3. Annihilation corresponds to **entropy saturation**, not modelling error.

Hence the annihilation operator ( \Pi_{\mathrm{ann}} ) is a **closure projector** on the ternary carrier.
