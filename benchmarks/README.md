# Benchmarks (Dense vs PQ)

Purpose: measure performance characteristics of dense vs PQ paths without gating CI. Results are JSONL for later plotting.

All outputs are timestamped automatically (suffix `-<suite>-YYYYMMDD-HHMMSS.jsonl`).

## Running (CPU)

```bash
# PQ encode/decode overhead
python benchmarks/bench.py --suite pq_roundtrip --sizes 1024 16384 65536 --sparsity 0.0 0.5 0.9 --repeats 5 --out benchmarks/results/pq_roundtrip.jsonl

# Dense kernel vs PQ roundtrip + kernel (sign flip)
python benchmarks/bench.py --suite kernel_dense_vs_pq --sizes 1024 16384 65536 --sparsity 0.0 0.5 0.9 --repeats 5 --out benchmarks/results/kernel_dense_vs_pq.jsonl

# PQ block-size sweep (auto block sizes; override with --blocks)
python benchmarks/bench.py --suite pq_block_sweep --sizes 1024 16384 65536 --sparsity 0.0 0.5 0.9 --blocks auto --repeats 3 --out benchmarks/results/pq_block_sweep.jsonl
```

Flags:
- `--sizes`: element counts
- `--sparsity`: fraction of zeros
- `--workload`: workload family (see `benchmarks/workloads.py`; e.g., `random_sparse`, `stencil_dense_iterated`, `alu_dense_burn`)
- `--blocks`: block sizes (integers) or `auto` (cache/workgroup heuristic) for block sweep
- `--repeats`: repeats per config
- `--seed`: RNG seed
- `--out`: JSONL output path prefix (timestamp is appended)
- `--batches`: for GPU suite, repeat the kernel this many times per timing (amortises dispatch); defaults to `1`
- `--iterations`: apply the kernel this many times inside each batch (increases arithmetic intensity)
- `--memory-mode`: informational hint (`host_visible` vs `device_local`) for how buffers are allocated/timed in Vulkan benchmarks

Notes:
- Benchmarks use dense `Carrier` semantics; PQ is measured as an optional storage/transport encoding.
- Correctness hashes are emitted per run; benchmarks are not CI-gating.
- Workload generators and axes live in `benchmarks/workloads.py`; the coverage map for public APIs lives in `benchmarks/coverage.yaml`. Run `python benchmarks/check_coverage.py` to ensure public exports are accounted for in benchmarking plans.

## Running (GPU dense, Vulkan sign-flip)

```bash
export VK_ICD_FILENAMES=/path/to/radeon_icd.x86_64.json  # adjust for your driver
glslc gpu_shaders/sign_flip.comp -o /tmp/sign_flip.spv
python benchmarks/bench.py --suite kernel_dense_vulkan --sizes 1024 16384 65536 --sparsity 0.0 0.5 0.9 --workload stencil_dense_iterated --iterations 10 --batches 1 4 8 --memory-mode host_visible --shader gpu_shaders/sign_flip.comp --spv /tmp/sign_flip.spv --device-index 0 --repeats 5 --out benchmarks/results/kernel_dense_vulkan.jsonl
```

Flags:
- `--shader`: GLSL compute shader (defaults to sign_flip.comp)
- `--spv`: SPIR-V output path (defaults to shader path with .spv)
- `--device-index`: Vulkan physical device index
- `--batches`: repeat the kernel this many times per timing (amortises dispatch)
- `--iterations`: apply the kernel this many times per batch (increase arithmetic intensity)
- `--memory-mode`: track which path was used in the benchmark plan (`host_visible` vs `device_local`)

Notes:
- Current Vulkan numbers are end-to-end wall times using host-visible buffers; dispatch/transfer overhead dominates for light kernels. Parity remains the priority; GPU speedups are workload-dependent.
- To measure compute-only for a GPU-favoured regime, keep data resident in device-local memory during the timed region, avoid map/unmap/readback inside the timing window, and do a single readback/hash after the loop. Report both host-visible and device-local modes explicitly via `--memory-mode`.
- Split metrics: `submit_to_fence_ms` for dispatch→fence, `wall_ms` for full run (readback/hash included). The Vulkan path should perform exactly one fence wait per timed run; per-dispatch waits will dominate and should be treated as a bug.
- A GPU-favoured workload is queued: `--workload alu_dense_burn` with large sizes (≥1M) and a `--rounds` axis to raise arithmetic intensity without changing semantics, timed in device-local mode with submit→fence timing. A rounds sweep (e.g., 1 vs 4096) should flip slope if compute is isolated.
- Latest runs (RX 580, sign_flip shader): host-visible vs device-local staging files are in `benchmarks/results/` (e.g., `kernel_dense_vulkan-stencil-host-visible-*.jsonl`, `kernel_dense_vulkan-stencil-device-local-*.jsonl`, `kernel_dense_vulkan-alu-host-visible-*.jsonl`, `kernel_dense_vulkan-alu-device-local-*.jsonl`). Device-local staging is functional but still overhead-dominated; compute-only timing remains TODO.
