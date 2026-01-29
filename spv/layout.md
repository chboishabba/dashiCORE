spv/
  core/
    # --- Fundamental algebra & state evolution ---
    operator_step.spv        # z_{t+1} = A z_t + b (+ optional nonlinearity)
    add.spv                  # elementwise add
    diff.spv                 # elementwise difference
    mul_scalar.spv           # x *= a
    axpy.spv                 # y = a*x + y
    fma3.spv                 # out = a*x + b*y + c
    clamp.spv                # bound values

    # --- Structural / sparse primitives ---
    push.spv                 # dst[idx[i]] = src[i]
    pop.spv                  # out[i] = src[idx[i]]
    scatter_add_atomic.spv   # dst[idx[i]] += src[i] (atomic)
    explode.spv              # replicate into multiple charts
    warp_affine_2d.spv       # per-tile affine warp
    warp_piecewise.spv       # piecewise (chart-based) warp

    # --- Reductions / invariants ---
    defect_local.spv         # per-site defect
    defect_reduce.spv        # scalar defect
    reduce_sum.spv
    reduce_max.spv
    reduce_sum_sq.spv
    norm_l2_finalize.spv

    # --- Dense / sparse linear operators ---
    gemv_tiled.spv           # dense GEMV baseline
    spmv_csr.spv             # sparse affine operator
    matmul_micro_8x8.spv     # small dense GEMM

    # --- Nonlinearities ---
    nonlinear_relu.spv
    nonlinear_tanh.spv
    nonlinear_sigmoid.spv
    nonlinear_admissibility_squash.spv

  pq/
    # --- PQ support/sign factorisation ---
    pq_support_sign.spv
    pq_pack_bits.spv
    pq_prefix_sum.spv
    pq_reconstruct.spv
    pq_mdl_cost.spv

    # --- PQ code streams ---
    bitpack_u4.spv
    bitunpack_u4.spv
    codes_unpack_from_u4.spv

    # --- PQ search ---
    pq_lut_build.spv
    pq_dist_accum_argmin.spv
    pq_dist_accum_argmin_u4.spv
    argmin_finalize.spv

  observer/
    qfeat.spv
    qfeat.fp64.spv
    stats_state.spv
    hash_state.spv

  io/
    nv12_to_r8.spv
    nv12_to_rgba.spv
    write_image.spv
    preview.vert.spv
    preview.frag.spv

docs/
  PQ_CODING.md
  pq_fs_layout.md
  kernel_taxonomy.md

  
  
  
  
  
  
  Perfect timing to do this cleanup. I’ll do **three things**, concisely and decisively:

1. **Update `spv/layout.md`** with a clean, future-proof structure
2. Produce the **one-page kernel taxonomy** (CORE vs PQ vs OBSERVER vs IO)
3. Answer the question **“do we prune for v1 or keep them all?”** with a clear recommendation

No fluff.

---

## 1️⃣ Updated `spv/layout.md`

This keeps **your existing structure**, adds what’s missing, and makes intent explicit.

```text
spv/
  core/
    # --- Fundamental algebra & state evolution ---
    operator_step.spv        # z_{t+1} = A z_t + b (+ optional nonlinearity)
    add.spv                  # elementwise add
    diff.spv                 # elementwise difference
    mul_scalar.spv           # x *= a
    axpy.spv                 # y = a*x + y
    fma3.spv                 # out = a*x + b*y + c
    clamp.spv                # bound values

    # --- Structural / sparse primitives ---
    push.spv                 # dst[idx[i]] = src[i]
    pop.spv                  # out[i] = src[idx[i]]
    scatter_add_atomic.spv   # dst[idx[i]] += src[i] (atomic)
    explode.spv              # replicate into multiple charts
    warp_affine_2d.spv       # per-tile affine warp
    warp_piecewise.spv       # piecewise (chart-based) warp

    # --- Reductions / invariants ---
    defect_local.spv         # per-site defect
    defect_reduce.spv        # scalar defect
    reduce_sum.spv
    reduce_max.spv
    reduce_sum_sq.spv
    norm_l2_finalize.spv

    # --- Dense / sparse linear operators ---
    gemv_tiled.spv           # dense GEMV baseline
    spmv_csr.spv             # sparse affine operator
    matmul_micro_8x8.spv     # small dense GEMM

    # --- Nonlinearities ---
    nonlinear_relu.spv
    nonlinear_tanh.spv
    nonlinear_sigmoid.spv
    nonlinear_admissibility_squash.spv

  pq/
    # --- PQ support/sign factorisation ---
    pq_support_sign.spv
    pq_pack_bits.spv
    pq_prefix_sum.spv
    pq_reconstruct.spv
    pq_mdl_cost.spv

    # --- PQ code streams ---
    bitpack_u4.spv
    bitunpack_u4.spv
    codes_unpack_from_u4.spv

    # --- PQ search ---
    pq_lut_build.spv
    pq_dist_accum_argmin.spv
    pq_dist_accum_argmin_u4.spv
    argmin_finalize.spv

  observer/
    qfeat.spv
    qfeat.fp64.spv
    stats_state.spv
    hash_state.spv

  io/
    nv12_to_r8.spv
    nv12_to_rgba.spv
    write_image.spv
    preview.vert.spv
    preview.frag.spv

docs/
  PQ_CODING.md
  pq_fs_layout.md
  kernel_taxonomy.md
```


### DASHI Kernel Taxonomy

#### **CORE kernels (semantic / mathematical)**

These define **the formal system**.
If a kernel changes the meaning of state, it is CORE.

**Categories**

* Algebra: `add`, `mul_scalar`, `axpy`, `fma3`
* Structure: `push`, `pop`, `explode`, `scatter_add`
* Geometry: `warp_affine`, `warp_piecewise`
* Dynamics: `operator_step`
* Invariants: `defect_*`, reductions, norms
* Linear operators: `gemv`, `spmv`, `matmul_micro`
* Nonlinear admissibility: `relu`, `tanh`, `sigmoid`, `squash`

**Rules**

* Deterministic unless explicitly marked atomic
* No IO, no format semantics
* Operates on abstract state vectors / sheets

CORE kernels are **non-negotiable** for correctness.

---

#### **PQ kernels (codec / compression layer)**

These implement **representation**, not dynamics.

**Responsibilities**

* Factorisation into support × sign
* Bit-packing / unpacking
* Distance approximation
* MDL accounting

**Rules**

* Must be invertible (within declared lattice)
* Must not leak into CORE semantics
* Can be swapped or upgraded independently

PQ kernels are **optional**, but complete as a subsystem.

---

#### **OBSERVER kernels (introspection / diagnostics)**

These do not affect state evolution.

**Examples**

* `qfeat`
* `stats_state`
* `hash_state`

Used for:

* debugging
* profiling
* validation
* visualization

Safe to remove in production builds.

---

#### **IO kernels (representation only)**

Pure format conversion or presentation.

**Examples**

* `nv12_to_rgba`
* `write_image`
* preview shaders

**Rule**

> IO kernels are *never* allowed to influence CORE logic.






## CORE-only minimal dispatch graph for `operator_step`

Assumptions:

* `operator_step.spv` is your “main” update kernel
* You want **GPU-only** execution and validation hooks
* Optional: nonlinearity, clamps, defect checking, norms/telemetry

### Graph A: absolute minimum (just step)

```
[ z_t ]  +  [A] (+ optional b)  ──►  operator_step  ──►  [ z_{t+1} ]
```

**Kernels:** `operator_step.spv`

That’s it. Everything else is optional.

---

### Graph B: minimal stable step (recommended default)

Adds boundedness and a sanity defect pass.

```
z_t ──► operator_step ──► z_tmp ──► clamp ──► z_{t+1}
                              │
                              └──► defect_local ──► reduce_sum OR reduce_max ──► defect_scalar
```

**Kernels used**

* Required: `operator_step.spv`
* Stability: `clamp.spv`
* Validation: `defect_local.spv` + (`reduce_sum.spv` **or** `reduce_max.spv`)

Notes:

* Use `reduce_max` if defect is “worst-case violation”
* Use `reduce_sum` if defect is “energy / total inconsistency”

---

### Graph C: nonlinear operator step (if you want “A then φ”)

```
z_t ──► operator_step ──► z_lin ──► nonlinear_* ──► z_nl ──► clamp ──► z_{t+1}
                                                      │
                                                      └─► (optional defect path)
```

**Kernels**

* `operator_step.spv`
* one of: `nonlinear_relu.spv` / `nonlinear_tanh.spv` / `nonlinear_sigmoid.spv` / `nonlinear_admissibility_squash.spv`
* `clamp.spv` (optional but strongly recommended for stability)
* optional defect path: `defect_local.spv` + reduction

---

### Graph D: residual / mixing form (common in practice)

If `operator_step` only computes `A z` and you want:
[
z_{t+1} = \alpha z_t + \beta (A z_t + b) + \gamma
]

```
z_t ──► operator_step ──► z_lin
  └──────────────────────────┐
                             ├─► fma3 (or axpy chain) ──► z_mix ──► (optional nonlinear/clamp)
consts ──────────────────────┘
```

**Kernels**

* `operator_step.spv`
* `fma3.spv` (preferred) or `axpy.spv` (+ `mul_scalar.spv`)
* optional: nonlinearity + clamp + defect checks

---

## Runtime capability matrix (kernels per feature flag)

Use this as a build/runtime registry: each flag implies a kernel set.
(Names are suggestions; adapt to your CLI/feature toggles.)

### Feature flags

#### `CORE_STEP`

**Meaning:** compute `z_{t+1}` on GPU

* **Required kernels:** `operator_step.spv`

---

#### `CORE_STEP_STABLE`

**Meaning:** step + enforce bounds

* **Requires:** `CORE_STEP`
* **Kernels:** `operator_step.spv`, `clamp.spv`

---

#### `CORE_NONLINEAR_{RELU|TANH|SIGMOID|SQUASH}`

**Meaning:** apply φ after step (or fused, if you later fuse)

* **Requires:** `CORE_STEP`
* **Kernels:** `operator_step.spv` + one of:

  * `nonlinear_relu.spv`
  * `nonlinear_tanh.spv`
  * `nonlinear_sigmoid.spv`
  * `nonlinear_admissibility_squash.spv`
* **Common add-on:** `clamp.spv` (recommended)

---

#### `CORE_VALIDATE_DEFECT_L1`

**Meaning:** scalar defect via sum reduction

* **Kernels:** `defect_local.spv`, `reduce_sum.spv` (repeat until 1 value), `defect_reduce.spv` *(if you keep defect_reduce as the final stage)*
* If `defect_reduce.spv` already performs the reduction chain internally, then just:

  * `defect_local.spv`, `defect_reduce.spv`

---

#### `CORE_VALIDATE_DEFECT_LINF`

**Meaning:** worst-case defect

* **Kernels:** `defect_local.spv`, `reduce_max.spv` (repeat), optional finalize

---

#### `CORE_STATS_L2`

**Meaning:** compute `||z||_2`

* **Kernels:** `reduce_sum_sq.spv`, `norm_l2_finalize.spv`

---

#### `CORE_STATS_HASH`

**Meaning:** hash state on GPU for parity checks (no readback of full buffers)

* **Kernels:** `hash_state.spv` *(observer-class but runtime flaggable)*

---

#### `CORE_STATS_SUMMARY`

**Meaning:** min/max/mean/sparsity telemetry

* **Kernels:** `stats_state.spv` *(observer-class but runtime flaggable)*

---

#### `CORE_SPARSE_AFFINE`

**Meaning:** apply sparse operator (CSR)

* **Kernels:** `spmv_csr.spv`
* Optional helpers depending on your layout: `gather_u32.spv`, `push.spv`, `pop.spv`

---

#### `CORE_DENSE_GEMV`

**Meaning:** apply dense operator (baseline)

* **Kernels:** `gemv_tiled.spv`

---

#### `CORE_DENSE_GEMM_MICRO`

**Meaning:** small dense matmul (when you need more than GEMV)

* **Kernels:** `matmul_micro_8x8.spv`

---

#### `CORE_STRUCTURAL_MAPS`

**Meaning:** index maps and chart manipulations

* **Kernels:** `push.spv`, `pop.spv`, `explode.spv`
* Optional: `scatter_add_atomic.spv` (if feature includes accumulation)

---

#### `CORE_ATOMIC_NONLINEAR_EXPLODE`

**Meaning:** allowed nondet atomic accumulation + nonlinearity + explode

* **Kernels:** either composed:

  * `explode.spv` + `nonlinear_*` + `scatter_add_atomic.spv`
    **or** fused (preferred when available):
  * `explode_nonlinear_scatter_atomic.spv`

---

## Suggested “minimal default profile” (fastest to ship)

Enable these by default:

* `CORE_STEP`
* `CORE_STEP_STABLE`
* optional: `CORE_VALIDATE_DEFECT_LINF` every N steps (cheap + robust)

