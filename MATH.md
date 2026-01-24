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
