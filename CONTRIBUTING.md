# dashiCORE — Core Operations & IO Specification

> **Status:** Canonical
> **Scope:** CPU reference semantics
> **Guarantee:** GPU / domain implementations must be observationally equivalent

---

## Conventions

* Let `Ω` denote an abstract index domain (lattice, graph, mesh, etc.)
* Let `T = {-1, 0, +1}` be the balanced ternary carrier
* Shapes are written using Python/NumPy-style ellipses (`[...]`)
* All operations must be **pure** unless explicitly stated

---

## 1. Carrier Operations

### 1.1 `Carrier.from_signed`

| Field         | Specification                                     |
| ------------- | ------------------------------------------------- |
| **Op**        | `Carrier.from_signed`                             |
| **Input**     | `signed: Int8[Ω]`, values ∈ `{-1,0,+1}`           |
| **Output**    | `Carrier(support: Bool[Ω], sign: Int8[Ω])`        |
| **Invariant** | `support[i] == (signed[i] != 0)`                  |
| **Invariant** | `sign[i] ∈ {-1,+1}` wherever `support[i] == True` |
| **Failure**   | Any value outside ternary set                     |

**Purpose:** Canonical factorisation of existence × orientation.

---

### 1.2 `Carrier.to_signed`

| Field         | Specification                  |
| ------------- | ------------------------------ |
| **Op**        | `Carrier.to_signed`            |
| **Input**     | `Carrier(support, sign)`       |
| **Output**    | `Int8[Ω] ∈ {-1,0,+1}`          |
| **Invariant** | Exact inverse of `from_signed` |

**Round-trip must be identity.**

---

### 1.3 `Carrier.validate`

| Field         | Specification            |
| ------------- | ------------------------ |
| **Op**        | `Carrier.validate`       |
| **Input**     | `Carrier`                |
| **Output**    | `None` or raises         |
| **Invariant** | Support/sign consistency |
| **Invariant** | No latent illegal states |

**This is a *hard* guardrail.**

---

## 2. Kernel Operations

### 2.1 `Kernel.apply`

| Field         | Specification                                      |
| ------------- | -------------------------------------------------- |
| **Op**        | `Kernel.apply`                                     |
| **Input**     | `state: Carrier[Ω]`, `ctx: AdmissibilityContext`   |
| **Output**    | `Carrier[Ω]`                                       |
| **Shape**     | Preserved exactly                                  |
| **Invariant** | Output carrier valid                               |
| **Invariant** | No new support created unless explicitly declared  |
| **Invariant** | Defect non-increasing (or documented monotonicity) |

**Kernel = local consistency projector.**

---

### 2.2 `Kernel.is_idempotent`

| Field       | Specification          |
| ----------- | ---------------------- |
| **Op**      | `Kernel.is_idempotent` |
| **Input**   | `state`                |
| **Output**  | `Bool`                 |
| **Meaning** | `K(K(x)) == K(x)`      |

Not all kernels must be idempotent, but they must **declare** if they are not.

---

## 3. Defect Operations

### 3.1 `Defect.local`

| Field         | Specification                   |
| ------------- | ------------------------------- |
| **Op**        | `Defect.local`                  |
| **Input**     | `pre: Carrier`, `post: Carrier` |
| **Output**    | `Float[Ω] ≥ 0`                  |
| **Invariant** | Zero iff locally consistent     |
| **Invariant** | Shape-aligned with carrier      |

---

### 3.2 `Defect.aggregate`

| Field         | Specification                               |
| ------------- | ------------------------------------------- |
| **Op**        | `Defect.aggregate`                          |
| **Input**     | `local_defect: Float[Ω]`                    |
| **Output**    | `Float ≥ 0`                                 |
| **Invariant** | Zero iff all local defects zero             |
| **Norm**      | Explicitly declared (`L1`, `L2`, max, etc.) |

Aggregation choice is **semantic**, not cosmetic.

---

### 3.3 `Defect.zero_test`

| Field       | Specification                      |
| ----------- | ---------------------------------- |
| **Op**      | `Defect.is_zero`                   |
| **Input**   | `state: Carrier`, `kernel: Kernel` |
| **Output**  | `Bool`                             |
| **Meaning** | Fixed point under kernel           |

---

## 4. Admissibility Operations

### 4.1 `Admissibility.apply`

| Field         | Specification                 |
| ------------- | ----------------------------- |
| **Op**        | `Admissibility.apply`         |
| **Input**     | `state: Carrier`, `transform` |
| **Output**    | `Carrier`                     |
| **Invariant** | Defect invariant              |
| **Invariant** | MDL invariant                 |

No admissibility transform may alter physical content.

---

### 4.2 `Admissibility.equivalent`

| Field       | Specification              |
| ----------- | -------------------------- |
| **Op**      | `Admissibility.equivalent` |
| **Input**   | `state_a`, `state_b`       |
| **Output**  | `Bool`                     |
| **Meaning** | Same equivalence class     |

This defines the **quotient space**.

---

## 5. MDL Operations

### 5.1 `MDL.score`

| Field         | Specification                      |
| ------------- | ---------------------------------- |
| **Op**        | `MDL.score`                        |
| **Input**     | `representation`                   |
| **Output**    | `Float ≥ 0`                        |
| **Invariant** | Lower = preferred                  |
| **Invariant** | Comparable across admissible forms |

MDL is **not** a loss function.

---

### 5.2 `MDL.compare`

| Field         | Specification                 |
| ------------- | ----------------------------- |
| **Op**        | `MDL.compare`                 |
| **Input**     | `rep_a`, `rep_b`              |
| **Output**    | `{A, B, tie}`                 |
| **Invariant** | Only admissible reps compared |

---

## 6. Hierarchy / M-Level Operations

### 6.1 `Lift.to_Mk`

| Field         | Specification                   |
| ------------- | ------------------------------- |
| **Op**        | `Lift.to_Mk`                    |
| **Input**     | `Carrier[M(n)]`                 |
| **Output**    | `Carrier[M(n+k)]`               |
| **Invariant** | Admissibility preserved         |
| **Invariant** | Defect composition well-defined |

---

### 6.2 `Project.to_lower`

| Field         | Specification               |
| ------------- | --------------------------- |
| **Op**        | `Project.to_lower`          |
| **Input**     | Higher-rank carrier         |
| **Output**    | Lower-rank carrier          |
| **Invariant** | No spurious defect creation |

---

## 7. Shape & Contract Summary (One-Glance)

| Category       | Shape Preserving | Defect Safe | Admissibility Invariant |
| -------------- | ---------------- | ----------- | ----------------------- |
| Carrier ops    | ✅                | N/A         | N/A                     |
| Kernel.apply   | ✅                | ✅           | ✅                       |
| Defect.compute | ✅                | N/A         | ✅                       |
| Admissibility  | ✅                | ✅           | ✅                       |
| MDL            | N/A              | N/A         | ✅                       |
| Lift / Project | ❌ (by design)    | ✅           | ✅                       |

---

## 8. Hard Rules (Non-Negotiable)

1. **No op may silently change shape**
2. **No op may create illegal ternary states**
3. **Defect zero defines fixed points**
4. **Admissibility is a quotient, not a mask**
5. **GPU backends must match these IO contracts exactly**

---

## 9. Acceptance Test Principle

> If two implementations differ **only** in backend,
> then for all admissible inputs they must:
>
> * Produce identical carriers
> * Produce identical defect scalars
> * Produce identical MDL ordering

Anything else is a bug.
