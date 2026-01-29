#!/usr/bin/env python3
"""Benchmark harness for dense vs PQ paths and block-size sweeps.

Outputs JSONL rows capturing timings and correctness hashes; does not gate CI.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from dashi_core.carrier import Carrier  # noqa: E402
from dashi_core.kernel import Kernel  # noqa: E402
from benchmarks.hardware import cpu_cache_info, recommend_pq_block_elems  # noqa: E402
from pq import decode_pq_to_carrier, encode_carrier_to_pq  # noqa: E402
from gpu_common_methods import compile_shader, resolve_shader, resolve_spv  # noqa: E402
from gpu_vulkan_backend import VulkanKernelConfig  # noqa: E402
from gpu_vulkan_dispatcher import DispatchTiming, VulkanCarrierDispatcher, VulkanDispatchConfig  # noqa: E402
import workloads  # noqa: E402


def _now_ms() -> float:
    return time.perf_counter_ns() / 1e6


def _hash_array(arr: np.ndarray) -> str:
    return str(np.int64(np.sum(arr.astype(np.int64) * 31)) % (10**12))


def _hash_gpu_style(sign: np.ndarray, support: np.ndarray) -> int:
    val = sign.astype(np.uint32) ^ (support.astype(np.uint32) << 1)
    mixed = val * np.uint32(2654435761)
    return int(np.bitwise_xor.reduce(mixed, dtype=np.uint32))


def _make_carrier(n: int, sparsity: float, seed: int, workload: str) -> Carrier:
    return workloads.make(workload, n, sparsity, seed)


class SignFlipKernel(Kernel):
    def apply(self, state: Carrier, ctx=None) -> Carrier:
        flipped = -state.sign
        return Carrier(support=state.support, sign=flipped)


@dataclass
class BenchResult:
    suite: str
    mode: str
    backend: str
    size: int
    sparsity: float
    batch_count: int
    block_elems: Optional[int]
    device: Optional[str]
    run: int
    memory_mode: Optional[str]
    t_encode_ms: float
    t_kernel_ms: float
    t_decode_ms: float
    t_total_ms: float
    t_gpu_compute_ms: Optional[float]
    t_submit_to_fence_ms: Optional[float]
    fence_waits: Optional[int]
    dispatches_per_run: Optional[int]
    hash_out: str
    hash_ref: str
    match: bool
    meta: dict

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def bench_pq_roundtrip(sizes: Sequence[int], sparsities: Sequence[float], repeats: int, seed: int, out: Path, workload: str):
    results: List[str] = []
    meta = {"cpu": cpu_cache_info()}
    for size in sizes:
        for sparsity in sparsities:
            carrier = _make_carrier(size, sparsity, seed, workload)
            ref_hash = _hash_array(carrier.to_signed())
            for run in range(repeats):
                start = _now_ms()
                buf = encode_carrier_to_pq(carrier)
                t_encode = _now_ms() - start
                start = _now_ms()
                decoded = decode_pq_to_carrier(buf)
                t_decode = _now_ms() - start
                match = np.array_equal(decoded.to_signed(), carrier.to_signed())
                res = BenchResult(
                    suite="pq_roundtrip",
                    mode="pq",
                    backend="cpu",
                    size=size,
                    sparsity=sparsity,
                    batch_count=1,
                    block_elems=None,
                    device=None,
                    run=run,
                    memory_mode=None,
                    t_encode_ms=t_encode,
                    t_kernel_ms=0.0,
                    t_decode_ms=t_decode,
                    t_total_ms=t_encode + t_decode,
                    t_gpu_compute_ms=None,
                    t_submit_to_fence_ms=None,
                    fence_waits=None,
                    dispatches_per_run=None,
                    hash_out=_hash_array(decoded.to_signed()),
                    hash_ref=ref_hash,
                    match=bool(match),
                    meta=meta,
                )
                results.append(res.to_json())
    _emit(out, results)


def bench_kernel_dense_vs_pq(
    sizes: Sequence[int],
    sparsities: Sequence[float],
    repeats: int,
    seed: int,
    workload: str,
    out: Path,
):
    results: List[str] = []
    meta = {"cpu": cpu_cache_info()}
    kernel = SignFlipKernel()
    for size in sizes:
        for sparsity in sparsities:
            carrier = _make_carrier(size, sparsity, seed, workload)
            ref_start = _now_ms()
            ref_out = kernel(carrier)
            ref_time = _now_ms() - ref_start
            ref_hash = _hash_array(ref_out.to_signed())
            for run in range(repeats):
                # Dense path timing
                dense_start = _now_ms()
                dense_out = kernel(carrier)
                dense_time = _now_ms() - dense_start
                dense_hash = _hash_array(dense_out.to_signed())

                # PQ path timing
                enc_start = _now_ms()
                buf = encode_carrier_to_pq(carrier)
                t_encode = _now_ms() - enc_start
                dec_start = _now_ms()
                decoded = decode_pq_to_carrier(buf)
                t_decode = _now_ms() - dec_start
                ker_start = _now_ms()
                pq_out = kernel(decoded)
                t_kernel = _now_ms() - ker_start
                pq_hash = _hash_array(pq_out.to_signed())
                res_dense = BenchResult(
                    suite="kernel_dense_vs_pq",
                    mode="dense",
                    backend="cpu",
                    size=size,
                    sparsity=sparsity,
                    batch_count=1,
                    block_elems=None,
                    device=None,
                    run=run,
                    memory_mode=None,
                    t_encode_ms=0.0,
                    t_kernel_ms=dense_time,
                    t_decode_ms=0.0,
                    t_total_ms=dense_time,
                    t_gpu_compute_ms=None,
                    t_submit_to_fence_ms=None,
                    fence_waits=None,
                    dispatches_per_run=None,
                    hash_out=dense_hash,
                    hash_ref=ref_hash,
                    match=bool(dense_hash == ref_hash),
                    meta=meta,
                )
                res_pq = BenchResult(
                    suite="kernel_dense_vs_pq",
                    mode="pq_roundtrip",
                    backend="cpu",
                    size=size,
                    sparsity=sparsity,
                    batch_count=1,
                    block_elems=None,
                    device=None,
                    run=run,
                    memory_mode=None,
                    t_encode_ms=t_encode,
                    t_kernel_ms=t_kernel,
                    t_decode_ms=t_decode,
                    t_total_ms=t_encode + t_decode + t_kernel,
                    t_gpu_compute_ms=None,
                    t_submit_to_fence_ms=None,
                    fence_waits=None,
                    dispatches_per_run=None,
                    hash_out=pq_hash,
                    hash_ref=ref_hash,
                    match=bool(pq_hash == ref_hash),
                    meta=meta,
                )
                results.append(res_dense.to_json())
                results.append(res_pq.to_json())
    _emit(out, results)


def _block_encode_decode_kernel(carrier: Carrier, block_elems: int, kernel: Kernel):
    n = carrier.sign.size
    t_encode = 0.0
    t_decode = 0.0
    t_kernel = 0.0
    outputs = []
    for start in range(0, n, block_elems):
        end = min(start + block_elems, n)
        sub = Carrier(support=carrier.support[start:end], sign=carrier.sign[start:end])
        s = _now_ms()
        buf = encode_carrier_to_pq(sub)
        t_encode += _now_ms() - s
        s = _now_ms()
        decoded = decode_pq_to_carrier(buf)
        t_decode += _now_ms() - s
        s = _now_ms()
        out = kernel(decoded)
        t_kernel += _now_ms() - s
        outputs.append((start, out))
    outputs = sorted(outputs, key=lambda t: t[0])
    support = np.concatenate([o.support for _, o in outputs])
    sign = np.concatenate([o.sign for _, o in outputs])
    return Carrier(support=support, sign=sign), t_encode, t_decode, t_kernel


def bench_pq_block_sweep(
    sizes: Sequence[int],
    sparsities: Sequence[float],
    blocks: Sequence[int],
    repeats: int,
    seed: int,
    workload: str,
    out: Path,
):
    results: List[str] = []
    meta = {"cpu": cpu_cache_info()}
    kernel = SignFlipKernel()
    for size in sizes:
        for sparsity in sparsities:
            carrier = _make_carrier(size, sparsity, seed, workload)
            ref_out = kernel(carrier)
            ref_hash = _hash_array(ref_out.to_signed())
            for block_elems in blocks:
                for run in range(repeats):
                    out_carrier, t_enc, t_dec, t_ker = _block_encode_decode_kernel(carrier, block_elems, kernel)
                    pq_hash = _hash_array(out_carrier.to_signed())
                    res = BenchResult(
                        suite="pq_block_sweep",
                        mode="pq_block",
                        backend="cpu",
                        size=size,
                        sparsity=sparsity,
                        batch_count=1,
                        block_elems=block_elems,
                        device=None,
                        run=run,
                        memory_mode=None,
                        t_encode_ms=t_enc,
                        t_kernel_ms=t_ker,
                        t_decode_ms=t_dec,
                        t_total_ms=t_enc + t_dec + t_ker,
                        t_gpu_compute_ms=None,
                        t_submit_to_fence_ms=None,
                        fence_waits=None,
                        dispatches_per_run=None,
                        hash_out=pq_hash,
                        hash_ref=ref_hash,
                        match=bool(pq_hash == ref_hash),
                        meta=meta,
                    )
                    results.append(res.to_json())
    _emit(out, results)


def _vulkan_device_info(device_index: int):
    try:
        import vulkan as vk
    except ImportError:
        return None
    instance = None
    try:
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="bench",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="bench",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_MAKE_VERSION(1, 2, 0),
        )
        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
        )
        instance = vk.vkCreateInstance(create_info, None)
        devices = vk.vkEnumeratePhysicalDevices(instance)
        if not devices or device_index >= len(devices):
            return None
        props = vk.vkGetPhysicalDeviceProperties(devices[device_index])
        raw_name = getattr(props, "deviceName", b"")
        if isinstance(raw_name, (bytes, bytearray)):
            try:
                name = raw_name.split(b"\x00", 1)[0].decode("utf-8", "replace")
            except Exception:
                name = str(raw_name)
        else:
            try:
                name = vk.string(raw_name)
            except Exception:
                name = str(raw_name)
        return {
            "device_name": name,
            "api_version": getattr(props, "apiVersion", None),
            "driver_version": getattr(props, "driverVersion", None),
        }
    except Exception:
        return None
    finally:
        if instance is not None:
            try:
                import vulkan as vk

                vk.vkDestroyInstance(instance, None)
            except Exception:
                pass


def bench_kernel_dense_vulkan(
    sizes: Sequence[int],
    sparsities: Sequence[float],
    repeats: int,
    seed: int,
    batches: Sequence[int],
    iterations: int,
    workload: str,
    out: Path,
    *,
    shader: Path,
    spv: Optional[Path],
    device_index: int,
    memory_mode: str = "host_visible",
    cpu_only: bool = False,
    gpu_only: bool = False,
    log_process_cpu: bool = False,
    gpu_hash_only: bool = False,
):
    """Benchmark dense Vulkan kernel (sign-flip) vs dense CPU reference."""
    if cpu_only and gpu_only:
        raise ValueError("cpu_only and gpu_only cannot both be set")
    if gpu_hash_only and cpu_only:
        raise ValueError("gpu_hash_only requires GPU path; cannot combine with cpu_only")
    if gpu_hash_only and memory_mode != "device_local":
        raise ValueError("gpu_hash_only requires device_local memory mode")

    results: List[str] = []
    meta = {
        "cpu": None if gpu_only else cpu_cache_info(),
        "vulkan": None,
        "iterations": iterations,
        "workload": workload,
        "memory_mode": memory_mode,
        "gpu_hash_only": gpu_hash_only,
    }

    dispatcher: Optional[VulkanCarrierDispatcher] = None
    shader = Path(shader)
    spv_path = Path(spv) if spv else resolve_spv(shader.stem)

    if not cpu_only:
        try:
            import vulkan as vk  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("python-vulkan not installed; cannot run GPU benchmarks") from exc
        meta["vulkan"] = _vulkan_device_info(device_index)
        if meta["vulkan"] is None:
            raise RuntimeError(
                "Vulkan device unavailable (python-vulkan present but no device enumerated). "
                "Set VK_ICD_FILENAMES or verify your driver installation before running GPU benchmarks."
            )
        compile_shader(shader, spv_path)
        config = VulkanKernelConfig(shader_path=shader, spv_path=spv_path, compile_on_dispatch=False)
        dispatch_config = VulkanDispatchConfig(device_index=device_index, memory_mode=memory_mode)
        dispatcher = VulkanCarrierDispatcher(config=config, dispatch_config=dispatch_config)

    ref_kernel = SignFlipKernel()

    for size in sizes:
        for sparsity in sparsities:
            carrier = _make_carrier(size, sparsity, seed, workload)
            for batch_count in batches:
                dispatches_per_run = batch_count * iterations
                ref_time = None
                ref_hash_dense = ""
                ref_out = None
                # Hash used for GPU-hash-only comparison (matches GPU reduction mix)
                ref_hash_gpu = str(_hash_gpu_style(carrier.sign, carrier.support))
                if not gpu_only:
                    ref_start = _now_ms()
                    for _ in range(dispatches_per_run):
                        ref_out = ref_kernel(carrier)
                    ref_time = _now_ms() - ref_start
                    if ref_out is not None:
                        ref_hash_dense = _hash_array(ref_out.to_signed())
                else:
                    # still compute reference hashes for GPU-only correctness check
                    ref_out = ref_kernel(carrier)
                    ref_hash_dense = _hash_array(ref_out.to_signed())
                for run in range(repeats):
                    meta_run = meta
                    if log_process_cpu:
                        cpu_pct = _process_cpu_pct()
                        meta_run = {**meta, "process_cpu_pct": cpu_pct}
                    if not gpu_only:
                        res_ref = BenchResult(
                            suite="kernel_dense_vulkan",
                            mode="cpu_dense_ref",
                            backend="cpu",
                            size=size,
                            sparsity=sparsity,
                            batch_count=batch_count,
                            block_elems=None,
                            device=None,
                            run=run,
                            memory_mode=None,
                            t_encode_ms=0.0,
                            t_kernel_ms=ref_time or 0.0,
                            t_decode_ms=0.0,
                            t_total_ms=ref_time or 0.0,
                            t_gpu_compute_ms=None,
                            t_submit_to_fence_ms=None,
                            fence_waits=None,
                            dispatches_per_run=dispatches_per_run,
                            hash_out=ref_hash_dense,
                            hash_ref=ref_hash_dense,
                            match=True,
                            meta=meta_run,
                        )
                        results.append(res_ref.to_json())

                    if not cpu_only:
                        vk_out, timing, hash_val = dispatcher.dispatch_batched(
                            carrier,
                            dispatches=dispatches_per_run,
                            collect_timing=True,
                            hash_only=gpu_hash_only,
                        )
                        submit_ms = timing.submit_to_fence_ms if timing else None
                        wall_ms = timing.wall_ms if timing else None
                        if gpu_hash_only:
                            vk_hash = str(hash_val) if hash_val is not None else ""
                            hash_ref = ref_hash_gpu
                            match = vk_hash == hash_ref
                        else:
                            vk_hash = _hash_array(vk_out.to_signed()) if vk_out is not None else ""
                            hash_ref = ref_hash_dense
                            match = bool(vk_hash == hash_ref)
                        res = BenchResult(
                            suite="kernel_dense_vulkan",
                            mode="vulkan_dense",
                            backend="vulkan",
                            size=size,
                            sparsity=sparsity,
                            batch_count=batch_count,
                            block_elems=None,
                            device=meta["vulkan"]["device_name"] if meta["vulkan"] else None,
                            run=run,
                            memory_mode=memory_mode,
                            t_encode_ms=0.0,
                            t_kernel_ms=wall_ms or 0.0,
                            t_decode_ms=0.0,
                            t_total_ms=wall_ms or 0.0,
                            t_gpu_compute_ms=submit_ms,
                            t_submit_to_fence_ms=submit_ms,
                            fence_waits=timing.fence_waits if timing else None,
                            dispatches_per_run=dispatches_per_run,
                            hash_out=vk_hash,
                            hash_ref=hash_ref,
                            match=match,
                            meta=meta_run,
                        )
                        results.append(res.to_json())
    _emit(out, results)


def _emit(out: Path, rows: Iterable[str]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        for line in rows:
            fh.write(line + "\n")


def _timestamped_path(path: Optional[Path], suite: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    if path is None:
        base = Path("benchmarks/results") / f"{suite}-{ts}.jsonl"
        return base
    if path.suffix:
        return path.with_name(f"{path.stem}-{suite}-{ts}{path.suffix}")
    return path / f"{suite}-{ts}.jsonl"


def _configure_threads(threads: Optional[int]) -> None:
    """Optionally constrain CPU threading for fair comparisons."""
    if threads is None or threads <= 0:
        return
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(threads)
    try:
        # numpy 1.25+ exposes set_num_threads for OpenBLAS/BLAS backends
        import numpy as _np

        if hasattr(_np, "set_num_threads"):
            _np.set_num_threads(threads)
    except Exception:
        pass


def _process_cpu_pct() -> Optional[float]:
    """Return instantaneous %CPU for current PID via ps, or None on failure."""
    try:
        out = subprocess.check_output(
            ["ps", "-p", str(os.getpid()), "-o", "%cpu", "--noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return float(out.strip().split()[0])
    except Exception:
        return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="dashiCORE benchmark runner (dense vs PQ).")
    p.add_argument(
        "--suite",
        choices=["pq_roundtrip", "kernel_dense_vs_pq", "pq_block_sweep", "kernel_dense_vulkan"],
        required=True,
    )
    p.add_argument("--sizes", type=int, nargs="+", default=[1024, 16384, 65536])
    p.add_argument("--sparsity", type=float, nargs="+", default=[0.0, 0.5, 0.9])
    p.add_argument("--blocks", type=str, nargs="*", help="Block sizes for pq_block_sweep (ints or 'auto')", default=None)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--workload", type=str, default="random_sparse", choices=list(workloads.names()))
    p.add_argument("--threads", type=int, default=None, help="Optional CPU thread cap (sets OMP/BLAS env + numpy.set_num_threads when available).")
    # Vulkan options
    p.add_argument("--shader", type=Path, default=resolve_shader("sign_flip"))
    p.add_argument("--spv", type=Path, default=None, help="Optional SPIR-V output path; defaults beside shader.")
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--batches", type=int, nargs="+", default=[1], help="Repeat kernel N times per timing (amortize dispatch).")
    p.add_argument("--iterations", type=int, default=1, help="Number of times to apply the kernel per batch (increase intensity).")
    p.add_argument(
        "--memory-mode",
        type=str,
        default="host_visible",
        choices=["host_visible", "device_local"],
        help="Benchmark memory mode hint (informational; device_local assumes staging outside timed region).",
    )
    p.add_argument("--cpu-only", action="store_true", help="For kernel_dense_vulkan, emit only CPU reference rows.")
    p.add_argument("--gpu-only", action="store_true", help="For kernel_dense_vulkan, emit only Vulkan rows (still requires a visible device).")
    p.add_argument("--log-process-cpu", action="store_true", help="Record instantaneous process %%CPU (via ps) into result meta.")
    p.add_argument("--gpu-hash-only", action="store_true", help="For kernel_dense_vulkan, skip full readback and use GPU-side hash reduction (device_local only).")
    return p.parse_args(argv)


def _parse_blocks(block_args: Optional[List[str]]) -> Optional[List[int]]:
    if not block_args:
        return None
    if len(block_args) == 1 and block_args[0] == "auto":
        info = cpu_cache_info()
        l1 = next((v for k, v in info.items() if k.startswith("L1") and "Data" in k), 32 * 1024)
        l2 = next((v for k, v in info.items() if k.startswith("L2")), 256 * 1024)
        line = info.get("line_size", 64)
        return recommend_pq_block_elems(l1_bytes=l1 or 32 * 1024, l2_bytes=l2 or 256 * 1024, cacheline_bytes=line or 64)
    blocks: List[int] = []
    for b in block_args:
        try:
            blocks.append(int(b))
        except ValueError:
            continue
    return sorted(set(blocks)) if blocks else None


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _configure_threads(args.threads)
    out = _timestamped_path(args.out, args.suite)
    if args.suite == "pq_roundtrip":
        bench_pq_roundtrip(args.sizes, args.sparsity, args.repeats, args.seed, out, args.workload)
    elif args.suite == "kernel_dense_vs_pq":
        bench_kernel_dense_vs_pq(args.sizes, args.sparsity, args.repeats, args.seed, args.workload, out)
    elif args.suite == "pq_block_sweep":
        blocks = _parse_blocks(args.blocks)
        if not blocks:
            blocks = recommend_pq_block_elems()
        bench_pq_block_sweep(args.sizes, args.sparsity, blocks, args.repeats, args.seed, args.workload, out)
    elif args.suite == "kernel_dense_vulkan":
        bench_kernel_dense_vulkan(
            args.sizes,
            args.sparsity,
            args.repeats,
            args.seed,
            args.batches,
            args.iterations,
            args.workload,
            out,
            shader=args.shader,
            spv=args.spv,
            device_index=args.device_index,
            memory_mode=args.memory_mode,
            cpu_only=args.cpu_only,
            gpu_only=args.gpu_only,
            log_process_cpu=args.log_process_cpu,
            gpu_hash_only=args.gpu_hash_only,
        )
    else:
        raise ValueError(f"Unknown suite: {args.suite}")


if __name__ == "__main__":
    main()
