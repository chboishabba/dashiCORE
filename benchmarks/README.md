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
- `--blocks`: block sizes (integers) or `auto` (cache/workgroup heuristic) for block sweep
- `--repeats`: repeats per config
- `--seed`: RNG seed
- `--out`: JSONL output path prefix (timestamp is appended)

Notes:
- Benchmarks use dense `Carrier` semantics; PQ is measured as an optional storage/transport encoding.
- Correctness hashes are emitted per run; benchmarks are not CI-gating.

## Running (GPU dense, Vulkan sign-flip)

```bash
export VK_ICD_FILENAMES=/path/to/radeon_icd.x86_64.json  # adjust for your driver
glslc gpu_shaders/sign_flip.comp -o /tmp/sign_flip.spv
python benchmarks/bench.py --suite kernel_dense_vulkan --sizes 1024 16384 65536 --sparsity 0.0 0.5 0.9 --shader gpu_shaders/sign_flip.comp --spv /tmp/sign_flip.spv --device-index 0 --repeats 5 --out benchmarks/results/kernel_dense_vulkan.jsonl
```

Flags:
- `--shader`: GLSL compute shader (defaults to sign_flip.comp)
- `--spv`: SPIR-V output path (defaults to shader path with .spv)
- `--device-index`: Vulkan physical device index
