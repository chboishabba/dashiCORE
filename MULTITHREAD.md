## 1. What “CPU multithread” means *in dashiCORE terms*

We are **not** talking about:

* shared-memory atomics
* unordered reductions
* implicit NumPy threading
* OpenMP magic

We **are** talking about:

> **Explicit data partitioning → independent kernel execution → deterministic recombination**

This mirrors the GPU mental model and keeps correctness proofs aligned.

---

## 2. Two distinct things (keep them separate)

### A. **Multithreaded execution**

> How kernels are actually run in parallel on CPU

### B. **Multithreaded benchmarks**

> Measuring when multithreading helps or hurts (dense vs PQ)

We should implement **A** first (execution), then **B** (measurement).

---

## 3. CPU multithread execution model (approved)

### Core rule

> **Threads never share writable state.**

Everything is:

* split
* computed independently
* recombined deterministically

---

### Execution strategy: block-parallel map

For a kernel `K` with locality radius `r = 0` (pointwise) or `r > 0` (stencil):

1. Partition carrier into blocks
2. (Optional) add halo of width `r`
3. Run `K` independently on each block
4. Crop halo
5. Recombine blocks in index order

This exactly matches the **parallelism tests** we defined earlier.

---

## 4. Threading backend choice (important)

### Use **`concurrent.futures`**, not NumPy threading

Why:

* Explicit control
* Deterministic ordering
* Easy benchmarking
* Works with pure Python kernels

### Default executor

```python
ThreadPoolExecutor
```

Why not `ProcessPoolExecutor` (yet)?

* `Carrier` copying cost
* IPC overhead
* Later option for large blocks

---

## 5. Minimal multithread kernel runner

This lives **outside CORE kernels**, e.g.:

```text
dashi_core/
  execution/
    cpu_parallel.py
```

### API sketch

```python
def run_kernel_parallel(
    kernel: Kernel,
    carrier: Carrier,
    *,
    num_workers: int,
    block_elems: int,
    halo: int = 0,
) -> Carrier:
    ...
```

---

### Pseudocode (deterministic)

```python
def run_kernel_parallel(kernel, carrier, num_workers, block_elems, halo=0):
    blocks = split_with_halo(carrier, block_elems, halo)

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [
            ex.submit(kernel, block)
            for block in blocks
        ]

        # IMPORTANT: preserve order
        results = [f.result() for f in futures]

    cores = crop_halo(results, halo)
    return recombine(cores)
```

Key properties:

* No shared writes
* No race conditions
* Same result as monolithic kernel (provable via existing tests)

---

## 6. CPU multithread **benchmarks** (what to measure)

Now we can benchmark **four modes**:

| Mode       | Description                       |
| ---------- | --------------------------------- |
| `dense-1t` | single-thread dense               |
| `dense-mt` | multithread dense                 |
| `pq-1t`    | PQ encode → decode, single-thread |
| `pq-mt`    | PQ encode/decode in parallel      |

Initially:

* kernels: sign-flip, clamp
* CPU only

---

## 7. Benchmark parameters (important)

### Sweep dimensions

* `N`: `[1e4, 1e5, 1e6]`
* `threads`: `[1, 2, 4, 8, cpu_count()]`
* `block_elems`: auto-selected + a few fixed
* `sparsity`: `[0.0, 0.5, 0.9]`

---

## 8. What we expect to learn (hypotheses)

These are *testable*:

### Dense path

* Wins for small N
* Scales until memory bandwidth saturates
* Likely best for branch-heavy kernels

### PQ path

* Encode/decode may parallelise well
* Wins only for large N
* Sensitive to block size
* Possible slowdown from cache thrashing

This will finally answer:

> “Does PQ + multithreading help, or is dense MT better?”

---

## 9. Determinism guarantees (must hold)

Even in multithread mode:

* Output **must equal** single-thread output
* Order of blocks **must not matter**
* Thread count **must not change result**

We already have the **parallelism tests** to enforce this.

---

## 10. Minimal next implementation (strongly recommended)

If you want the *smallest high-impact step*, do this:

### Step 1

Implement:

```text
dashi_core/execution/cpu_parallel.py
```

with:

* `run_kernel_parallel`
* block splitting
* recombination

### Step 2

Add **one test**:

```text
tests/kernel/test_kernel_parallel_cpu.py
```

Assertion:

```python
K(x) == run_kernel_parallel(K, x, threads=4)
```

### Step 3

Add benchmark mode:

```bash
bench.py --mode dense-mt
```

That’s it — no PQ yet.

---

## 11. Why this matters strategically

Once this exists:

* GPU = just another parallel backend
* Multi-node later = same abstraction
* PQ becomes *optional optimisation*, not architectural commitment
* Performance decisions become data-driven

You’re essentially building a **deterministic parallel algebra** over kernels.
