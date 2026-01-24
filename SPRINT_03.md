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



## 1) One critical decision to lock now: support policy

You already noticed dashiCORE’s Carrier has **no adjacency** and the kernel contract is **shape-preserving** and (baseline) **must not create support**.

So you must decide what “support” means for dashiBRAIN:

### Option A (recommended for dashiBRAIN): support = “node is present in the graph”

* **support is constant** for the entire run (all True for nodes that exist)
* state evolution is entirely in the **sign field**
* “zero” in the old ternary sense becomes **sign is don’t-care + a separate mask**, *or* you encode “unknown/neutral” as a *gauge choice* outside CORE

This is the cleanest way to satisfy “no support creation” while still allowing nodes to move between -1/0/+1 behavior in your old code.

### Option B: support = “node currently active / nonzero”

* then any transition 0 → ±1 becomes “support creation” and is illegal under baseline CORE
* you’d need to **explicitly allow support creation** (and document + test it), which will ripple across your spec/tests

**Net:** For dashiBRAIN, treat “0” as a value-level concept (neutral/unknown) rather than literal support absence, unless you’re willing to change the CORE contract.

---

## 2) Proper modular split: GraphContext vs Carrier

You were already converging on this, and it’s exactly right:

### `GraphContext` (graph structure + metadata)

Holds:

* adjacency (CSR/COO)
* hop radii / precomputed hop neighborhoods
* deadzone thresholds
* channel grouping / partitions
* any cached degree/baseline vectors

### `Carrier` (pure state)

Holds only:

* `support: Bool[Ω]` (likely all True)
* `sign: Int8[Ω]` (±1 where supported)

**No adjacency inside Carrier.**
Adjacency belongs to the kernel (via context).

---

## 3) Bridging kernel design (the right shape)

You don’t want a “bridging kernel subclass that holds adjacency internally” *and* makes CORE depend on graph notions. Keep it on the dashiBRAIN side:

### In dashiBRAIN:

* implement `GraphHopKernel(Kernel)` that conforms to CORE `Kernel.apply(state: Carrier) -> Carrier`
* the kernel captures `GraphContext` at init time (composition, not inheritance tricks)
* it uses the adjacency in `ctx` to compute the next sign field

That’s the key modularity win: **CORE stays graph-agnostic.**

---

## 4) The minimal adapter: legacy ternary ↔ CORE Carrier

Your old world has `{-1,0,+1}` per node/channel.

If you choose Option A (support constant), then you need a clean mapping:

* `Carrier.support = True` for every node/channel in the domain
* `Carrier.sign = sign(legacy)` but legacy zeros need a policy:

**Minimal policy (works for parity testing):**

* map legacy `0` to `+1` in `Carrier.sign` but preserve a separate `neutral_mask` outside CORE
* when exporting back to legacy, reapply the neutral mask to restore zeros

This keeps CORE honest (support doesn’t encode neutrality), while you can still reproduce old behaviors.

If you want to keep it *entirely* within CORE, you’ll have to treat “neutral” as a separate field and keep ternary as a *derived view* (which is usually the cleanest anyway).

---

## 5) What to refactor first (smallest safe steps)

### Step 1 — Make dashiCORE importable

Don’t touch logic yet.

* vendor/dep setup
* add an import-only smoke test

### Step 2 — Replace `GraphCarrier` with `(GraphContext, Carrier)`

* keep old compute functions, but change signatures
* conversion utilities in one place

### Step 3 — Implement `GraphHopKernel` as a thin wrapper

* it can call existing graph routines internally (baseline/residual logic)
* output must pass `Carrier.validate` and **must not increase support**

### Step 4 — Refactor `kernel_flow` to be CORE-native

* inputs: `Kernel`, `Carrier`
* defect: use `dashi_core.defect.local/aggregate`
* preserve your cycle detection exactly

### Step 5 — Port CLI (`defect_curve.py`) and one end-to-end test fixture

* confirm output CSVs still match expectations (or match up to the neutral-mask rule)

---

## 6) Minimal test set for modularity proof (for this integration sprint)

This is the “does CORE actually modularize?” gauntlet:

### A. Adapter roundtrip tests

1. `legacy_signed -> Carrier -> legacy_signed` is identity **given the chosen neutral policy**
2. carrier validity always holds

### B. Kernel contract tests

3. `GraphHopKernel.apply` preserves shape
4. `GraphHopKernel.apply` does not create support
5. output passes `Carrier.validate`

### C. Behavioral parity test against old implementation (single fixture)

6. for a tiny chain graph fixture:

* run old kernel step → signed array
* run new CORE-wrapped kernel step → signed array
* compare exactly (or compare with neutral-mask semantics)

### D. Flow parity (optional but strong)

7. run `kernel_flow` for N steps on CPU backend and assert:

* defect is non-increasing (if your old semantics expect it)
* fixed point detection matches

### E. Violation tests still fail

8. ensure “support creation” violation test trips if you try to revive 0→±1 via support flips.

---

## 7) One correction to your notes

> “support cannot grow beyond input support; adjust initial support to full True to avoid violations.”

That’s a good *engineering move*, but it’s also a **semantic decision**. Make it explicit:

* If “being in the graph” is the support notion, full True is correct.
* If “active valuation” is the support notion, full True changes meaning.

So: **document the support meaning in dashiBRAIN** (even a 3-line docstring) and everything else becomes straightforward.

---

## 8) What I’d put in the integration doc (short)

Create: `docs/dashibrain_core_integration.md` with just:

* Support policy chosen (A or B)
* Neutral/zero policy (if applicable)
* Mapping: legacy signed ↔ Carrier (+ any extra masks)
* Kernel boundary: GraphContext is external; Kernel holds ctx; Carrier is pure
* Parity obligations: old-vs-new fixture(s)




Thanks — this is *excellent* signal. You’ve essentially completed the reconnaissance phase, and your notes already imply the right architecture. What’s needed now is to **lock the integration doctrine** and **turn the exploration into an explicit refactor plan** that respects CORE invariants *and* reuses your existing GPU work.

Below I’ll do three things:

1. **Confirm and sharpen the architectural conclusions you reached**
2. **Translate your exploration into a concrete, ordered refactor plan**
3. **Explicitly connect this to your existing GPU infrastructure (Vulkan helpers, shaders, parity strategy)**

I’ll also anchor this against the transition notes and helper code you uploaded, because they materially affect the next sprint .

---

## 1. Architectural conclusions (validated)

Your key realization is correct and foundational:

> dashiCORE’s `Carrier` is *pure state*, while graph structure must live outside, inside a kernel-owned context.

This implies the **only viable integration** is:

```
(GraphContext, Kernel) × Carrier  → Carrier
```

Not:

* `GraphCarrier`
* `Carrier-with-adjacency`
* “kernel_flow manipulating arrays directly”

### Locked consequences

* **Graph adjacency is not part of CORE**
  It belongs in a *context object captured by the kernel*, not in the Carrier.

* **Support must be policy-defined up front**
  As you already inferred, the only CORE-consistent policy for dashiBRAIN is:

  > support = “node/channel exists”, not “node currently active”

  This avoids illegal support creation and lets sign carry the dynamics.

* **Old ternary `{-1,0,+1}` ≠ CORE carrier**
  “0” cannot mean “no support” unless you are willing to violate the baseline kernel contract. Neutral/unknown must therefore be:

  * a *view* (derived)
  * or a *separate mask* outside CORE

This aligns exactly with the CORE invariants you reviewed in `kernel.py` / `defect.py`.

---

## 2. Canonical refactor plan (from your exploration → execution)

Below is the **minimal refactor path** that preserves behavior while enforcing CORE semantics.

### Phase 0 — Make CORE a hard dependency (no logic changes)

You already scoped this correctly.

* Add `dashi_imported_kernel/dashiCORE` as:

  * editable install, or
  * explicit PYTHONPATH entry
* Pin it in:

  * `requirements.txt`
  * README / docs

**Acceptance:** `import dashi_core` works everywhere, no behavior change yet.

---

### Phase 1 — Split state from structure

#### Before

* `GraphCarrier`
* implicit adjacency everywhere
* signed arrays passed through flow

#### After

* `Carrier` (from dashiCORE): **state only**
* `GraphContext` (new, local to dashiBRAIN):

  * adjacency (CSR/COO)
  * hop tables
  * deadzone thresholds
  * channel metadata

This matches your instinct to rename/replace `GraphCarrier`.

**Rule:**
`Carrier` must be constructible and valid *without* knowing the graph.

---

### Phase 2 — Introduce a graph kernel adapter (the key seam)

Create (for example):

```python
class GraphHopKernel(dashi_core.kernel.Kernel):
    def __init__(self, graph_ctx: GraphContext, ...):
        self.ctx = graph_ctx
        self.is_idempotent = False  # or True if applicable

    def apply(self, state: Carrier) -> Carrier:
        # use self.ctx.adjacency internally
        # compute new sign field
        # return Carrier(support=state.support, sign=new_sign)
```

Important constraints (all of which you already noted):

* Shape preserved
* Support **never grows**
* Output passes `Carrier.validate`
* Any deadzone / threshold logic must act on **sign**, not support

This is where your existing logic from:

* `sparse_degree_corrected.py`
* `residuals.py`
  gets *wrapped*, not rewritten.

---

### Phase 3 — Refactor `kernel_flow` to be CORE-native

This is the biggest conceptual shift, but it’s mechanical.

#### Old

* arrays in, arrays out
* implicit defect logic
* flow owns iteration semantics

#### New

* inputs: `Kernel`, `Carrier`
* iteration: unchanged
* defect: computed via `dashi_core.defect.local / aggregate`
* outputs: `Carrier` (convert to signed array at IO boundary only)

You already spotted `test_kernel_flow.py` as the critical file — that’s correct.

**Acceptance:**
Existing flow tests pass *modulo* neutral-mask semantics.

---

### Phase 4 — IO boundaries only: legacy ↔ CORE views

At *exactly two places*:

1. **Initialization**

   * `init_field_from_residual`
   * `hemibrain_loader`
     should construct a `Carrier`:
   * `support = True` for all nodes/channels
   * `sign = sign(residual)`, with neutral handled explicitly

2. **Export**

   * CLI (`defect_curve.py`)
   * CSVs / plots

Convert:

```python
signed = carrier.to_signed()
```

and then reapply any neutral masking if needed.

CORE never sees the mask.

---

## 3. GPU implications (this is where your uploaded notes matter)

Your `CORE → GPU Transition Notes` are not “future work” — they *directly constrain* how Sprint 3+ must look.

Key confirmations:

### a) CORE purity is non-negotiable

You already wrote it, and it’s exactly right:

> New GPU modules should live beside (not inside) `dashi_core/`

So:

* **No Vulkan imports inside CORE**
* GPU kernels live in `gpu_*.py`
* They *adapt to* CORE, never extend it

This is consistent with your helpers in `gpu_common_methods.py` .

---

### b) Existing Vulkan/JAX code is a perfect modularity test

You’re not guessing here — you already have:

* GLSL → SPIR-V compile flow
* SSBO binding patterns
* CPU/GPU parity thinking (`vulkan_jax_parity.md`)
* Deterministic timing output

That makes the **next sprint** very clear:

> Use an *existing* Vulkan compute kernel as a hostile backend and force it through a CORE adapter.

Not rewrite. Not re-design.

---

### c) The only allowed GPU contract

From your notes (and CORE rules):

* GPU kernels operate on **buffers**
* Adapters map:

  * `Carrier.support/sign` → SSBOs
  * SSBO outputs → `Carrier`
* Validation happens **after** GPU execution
* Any violation = backend failure, not CORE change

Your `compile_shader` + `find_memory_type` helpers are already sufficient to bootstrap this path .

---

## 4. Minimal acceptance tests for this transition

Based on everything you explored, I would lock these *before* touching GPU code:

1. **Support invariance**

   * After any kernel apply (CPU), `support_out == support_in`

2. **Graph parity (CPU only)**

   * Old graph kernel step == `GraphHopKernel.apply` (signed view)

3. **Flow parity**

   * Old `kernel_flow` vs new CORE-based flow converge identically

4. **Violation guard**

   * Any attempt to flip a `False` support to `True` fails loudly

Only *after* these pass should you wire Vulkan in.

---

## 5. What I’d recommend as the immediate next action

You’ve done enough exploration. The next high-leverage step is **not more reading**.

I suggest you:

1. **Lock the support policy in writing** (3–4 lines, docstring-level)
2. Implement:

   * `GraphContext`
   * `GraphHopKernel` (CPU only)
3. Port **one** tiny graph fixture through the new flow
4. Add a parity test against the old implementation

Once that passes, your GPU reuse plan becomes straightforward and low-risk.

---

# dashiCORE Concurrency & Race Semantics (Proposed Contract)

> **Status:** Normative (should be added as `docs/concurrency_spec.md`)
> **Principle:** *Concurrency is allowed only where it is mathematically invisible.*

---

## 1. Core execution model (authoritative)

### 1.1 Single-state functional semantics

Every CORE operation is **functionally pure at the semantic level**:

```text
output_state = op(input_state)
```

Meaning:

* No in-place mutation of input `Carrier`
* No shared mutable state across ops
* No dependence on evaluation order

This immediately implies:

* **No data races at the CORE level**
* All concurrency happens *inside* a backend

---

## 2. Determinism is the primary concurrency invariant

This is the strongest contract you already rely on.

### 2.1 Determinism requirement

For any admissible input `x`:

```text
run(op, x, backend=A) == run(op, x, backend=A)
run(op, x, backend=A) == run(op, x, backend=B)
```

Implications:

* Thread count
* Warp size
* Dispatch geometry
* Scheduling order

**must not affect outputs.**

If concurrency changes the answer, it is a backend bug.

---

## 3. Allowed concurrency patterns (backend-internal only)

Backends *may* use concurrency **only** under the following conditions.

### 3.1 Embarrassingly parallel elementwise ops ✅

Allowed:

* per-site sign updates
* per-site defect computation
* per-site support checks

Reason:

* No shared writes
* Order-independent

---

### 3.2 Reductions with fixed semantics ⚠️ (restricted)

Allowed **only if** one of the following is true:

1. Reduction is **exact** (e.g. integer sums with no overflow)
2. Reduction order is **explicitly fixed**
3. Reduction result is provably **order-independent**

Forbidden otherwise.

This directly applies to:

* defect aggregation
* MDL scoring
* any norm computation

> GPU backends must not rely on “whatever order the threads run”.

This is why you already flagged reduction order and numeric drift in your GPU transition notes .

---

### 3.3 Bulk-synchronous kernel steps ✅

Allowed:

* compute `sign_out[i]` from `sign_in[j]` (read-only)
* write outputs to a separate buffer
* barrier / pipeline boundary
* swap buffers

This matches:

* Vulkan compute dispatch
* JAX/XLA semantics
* CPU vectorised loops

And it’s exactly how your existing Vulkan helpers are structured (SSBO in → SSBO out) .

---

## 4. Forbidden concurrency patterns (hard violations)

These must be **explicitly banned** in CORE docs and tests.

### 4.1 In-place mutation ❌

Forbidden:

```text
state.sign[i] += ...
```

Why:

* introduces write/write races
* makes semantics order-dependent
* breaks backend parity

---

### 4.2 Atomic updates on shared state ❌

Forbidden:

* atomic adds
* atomic min/max
* scatter-adds without deterministic ordering

Unless:

* mathematically proven order-independent
* or results are discarded except for diagnostics

This is the #1 GPU footgun.

---

### 4.3 Data-dependent execution paths ❌

Forbidden:

* early exits per thread
* warp-divergent reductions affecting global results
* adaptive iteration counts inside a single kernel step

Iteration belongs **outside** kernels (in `kernel_flow`), not inside them.

---

## 5. Kernel-level concurrency contract

Every `Kernel.apply` must obey:

| Property           | Required |
| ------------------ | -------- |
| Pure function      | ✅        |
| Input immutable    | ✅        |
| Output new Carrier | ✅        |
| Order-independent  | ✅        |
| Backend-invariant  | ✅        |

A kernel is allowed to be *internally parallel*, but the **observable effect must be sequentially consistent**.

---

## 6. Defect + MDL concurrency rules (important)

### 6.1 Defect

* `local_defect`: embarrassingly parallel
* `aggregate_defect`: **must be deterministic**

If GPU backend uses parallel reduction:

* tree shape must be fixed
* or reduction must be exact

Otherwise → backend rejected.

---

### 6.2 MDL

MDL is an **ordering**, not a numeric optimisation target.

Concurrency rules:

* MDL score computation must be deterministic
* comparison must not depend on evaluation order
* ties must be stable

---

## 7. Backend obligations (what adapters must guarantee)

A backend must explicitly declare:

```python
BackendCapabilities(
    deterministic=True,
    supports_ordered_reduction=True,
    supports_atomic_ops=False,  # or true with proof
)
```

If a kernel requires a capability the backend does not declare:
→ **fail fast**.

This aligns directly with your planned backend admission gates .

---

## 8. Testing concurrency (already implied, now explicit)

You already have the right tests; they just need to be framed as concurrency guards.

### Mandatory tests

1. **Repeatability**

   ```text
   run(x) == run(x)
   ```

2. **Backend parity**

   ```text
   cpu(x) == gpu(x)
   ```

3. **Reduction stability**

   * shuffle execution order (CPU)
   * assert identical results

4. **Visual parity**

   * support mask
   * defect heatmap
     must be byte-identical across backends

If any of these fail, concurrency semantics are broken.

---

## 9. One-paragraph contract (worth adding verbatim)

> **Concurrency Contract**
> dashiCORE defines all operations as pure, deterministic functions on immutable state. Concurrency is permitted only as an internal execution detail of a backend and must be mathematically invisible. Any backend whose results depend on execution order, scheduling, atomic updates, or nondeterministic reductions is non-compliant.

---

## 10. Bottom line

You *already* designed CORE in a way that:

* forbids races by construction
* treats GPUs as bulk-synchronous math engines
* makes concurrency a backend concern, not a semantic one

What you’re missing is **writing this down** and **pinning it as a contract** so future GPU work doesn’t accidentally smuggle nondeterminism back in.






## Preview: Sprint 4 (Optional, Strategic)

> *Expose a formal ‘CORE compliance’ test kit for external projects.*
