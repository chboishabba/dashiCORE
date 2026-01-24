# PQ vs Dense Policy (Execution & Storage)

## Canonical rule
- Dense `Carrier` is the semantic truth for dashiCORE.
- PQ is an optional storage/transport encoding and must be observationally invisible to kernel semantics.

## When to use each
- Dense (default): all kernels, CPU reference paths, correctness/parity/invariant tests, branching/parallelism tests, defect/admissibility reasoning.
- PQ (opt-in): filesystem checkpoints, logging, large memory-bound transfers where bandwidth dominates, benchmarking PQ vs dense. Never used directly by kernels.

## Why PQ can be slower
- Small problems: encode/decode overhead dominates.
- Compute- or branch-heavy kernels: packing adds instructions and divergence.
- Poor block sizing: too small → overhead; too large → cache/register pressure.

## Comparison policy
- Every PQ path must have a dense twin; dense remains the baseline.
- PQ is enabled only when data shows a win (size threshold + memory-bound + encode/decode < ~20–30% total).
- Correctness comparisons always decode to dense before checking equality.

## Benchmark guidance (future work)
- Benchmark dimensions: size sweep, sparsity sweep, backend (CPU/Vulkan), representation (dense/PQ), block size.
- Metrics: encode/decode time, kernel time, total time, bytes moved, effective bandwidth, hashes for correctness.
- Output format: JSONL rows per run for plotting crossover points.

## Current state
- Implemented: dense kernels and GPU parity tests; PQ encode/decode helpers (`pq.py`) with roundtrip tests (`tests/pq/`).
- Not implemented yet: PQ-backed GPU buffers, PQ vs dense benchmark harness, automatic PQ block-size selection.

## One-sentence summary
Dense Carrier execution is the canonical baseline in dashiCORE. PQ encoding is maintained as an optional storage/transport optimisation and is evaluated empirically against dense execution; both paths are preserved and PQ is used selectively where benchmarks demonstrate a clear advantage.
