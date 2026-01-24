#!/usr/bin/env python3
"""Benchmark harness for dense vs PQ paths and block-size sweeps.

Outputs JSONL rows capturing timings and correctness hashes; does not gate CI.
"""

from __future__ import annotations

import argparse
import json
import os
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

from dashi_core.carrier import Carrier  # noqa: E402
from dashi_core.kernel import Kernel  # noqa: E402
from benchmarks.hardware import cpu_cache_info, recommend_pq_block_elems  # noqa: E402
from pq import decode_pq_to_carrier, encode_carrier_to_pq  # noqa: E402
from gpu_common_methods import compile_shader  # noqa: E402
from gpu_vulkan_backend import make_vulkan_kernel, register_vulkan_backend, VulkanKernelConfig  # noqa: E402
from gpu_vulkan_dispatcher import VulkanDispatchConfig  # noqa: E402


def _now_ms() -> float:
    return time.perf_counter_ns() / 1e6


def _hash_array(arr: np.ndarray) -> str:
    return str(np.int64(np.sum(arr.astype(np.int64) * 31)) % (10**12))


def _make_carrier(n: int, sparsity: float, seed: int) -> Carrier:
    rng = np.random.default_rng(seed)
    probs = [sparsity, (1 - sparsity) / 2, (1 - sparsity) / 2]
    vals = rng.choice([0, -1, 1], size=n, p=probs).astype(np.int8)
    return Carrier.from_signed(vals)


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
    block_elems: Optional[int]
    device: Optional[str]
    run: int
    t_encode_ms: float
    t_kernel_ms: float
    t_decode_ms: float
    t_total_ms: float
    hash_out: str
    hash_ref: str
    match: bool
    meta: dict

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def bench_pq_roundtrip(sizes: Sequence[int], sparsities: Sequence[float], repeats: int, seed: int, out: Path):
    results: List[str] = []
    meta = {"cpu": cpu_cache_info()}
    for size in sizes:
        for sparsity in sparsities:
            carrier = _make_carrier(size, sparsity, seed)
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
                    block_elems=None,
                    device=None,
                    run=run,
                    t_encode_ms=t_encode,
                    t_kernel_ms=0.0,
                    t_decode_ms=t_decode,
                    t_total_ms=t_encode + t_decode,
                    hash_out=_hash_array(decoded.to_signed()),
                    hash_ref=ref_hash,
                    match=bool(match),
                    meta=meta,
                )
                results.append(res.to_json())
    _emit(out, results)


def bench_kernel_dense_vs_pq(sizes: Sequence[int], sparsities: Sequence[float], repeats: int, seed: int, out: Path):
    results: List[str] = []
    meta = {"cpu": cpu_cache_info()}
    kernel = SignFlipKernel()
    for size in sizes:
        for sparsity in sparsities:
            carrier = _make_carrier(size, sparsity, seed)
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
                    block_elems=None,
                    device=None,
                    run=run,
                    t_encode_ms=0.0,
                    t_kernel_ms=dense_time,
                    t_decode_ms=0.0,
                    t_total_ms=dense_time,
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
                    block_elems=None,
                    device=None,
                    run=run,
                    t_encode_ms=t_encode,
                    t_kernel_ms=t_kernel,
                    t_decode_ms=t_decode,
                    t_total_ms=t_encode + t_decode + t_kernel,
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
    out: Path,
):
    results: List[str] = []
    meta = {"cpu": cpu_cache_info()}
    kernel = SignFlipKernel()
    for size in sizes:
        for sparsity in sparsities:
            carrier = _make_carrier(size, sparsity, seed)
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
                    block_elems=block_elems,
                    device=None,
                    run=run,
                    t_encode_ms=t_enc,
                    t_kernel_ms=t_ker,
                    t_decode_ms=t_dec,
                    t_total_ms=t_enc + t_dec + t_ker,
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
        return {"device_name": vk.string(props.deviceName), "api_version": props.apiVersion}
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
    out: Path,
    *,
    shader: Path,
    spv: Optional[Path],
    device_index: int,
):
    """Benchmark dense Vulkan kernel (sign-flip) vs dense CPU reference."""
    results: List[str] = []
    meta = {"cpu": cpu_cache_info(), "vulkan": _vulkan_device_info(device_index)}
    try:
        import vulkan as vk  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("python-vulkan not installed; cannot run Vulkan benchmarks") from exc

    shader = Path(shader)
    spv_path = Path(spv) if spv else shader.with_suffix(".spv")
    compile_shader(shader, spv_path)

    config = VulkanKernelConfig(shader_path=shader, spv_path=spv_path, compile_on_dispatch=False)
    backend = register_vulkan_backend(
        name="bench_vulkan_dense",
        config=config,
        dispatch_config=VulkanDispatchConfig(device_index=device_index),
        allow_fallback=False,
    )
    kernel = make_vulkan_kernel(backend)
    ref_kernel = SignFlipKernel()

    for size in sizes:
        for sparsity in sparsities:
            carrier = _make_carrier(size, sparsity, seed)
            ref_start = _now_ms()
            ref_out = ref_kernel(carrier)
            ref_time = _now_ms() - ref_start
            ref_hash = _hash_array(ref_out.to_signed())
            for run in range(repeats):
                start = _now_ms()
                vk_out = kernel(carrier)
                t_kernel = _now_ms() - start
                vk_hash = _hash_array(vk_out.to_signed())
                res = BenchResult(
                    suite="kernel_dense_vulkan",
                    mode="vulkan_dense",
                    backend="vulkan",
                    size=size,
                    sparsity=sparsity,
                    block_elems=None,
                    device=meta["vulkan"]["device_name"] if meta["vulkan"] else None,
                    run=run,
                    t_encode_ms=0.0,
                    t_kernel_ms=t_kernel,
                    t_decode_ms=0.0,
                    t_total_ms=t_kernel,
                    hash_out=vk_hash,
                    hash_ref=ref_hash,
                    match=bool(vk_hash == ref_hash),
                    meta=meta,
                )
                res_ref = BenchResult(
                    suite="kernel_dense_vulkan",
                    mode="cpu_dense_ref",
                    backend="cpu",
                    size=size,
                    sparsity=sparsity,
                    block_elems=None,
                    device=None,
                    run=run,
                    t_encode_ms=0.0,
                    t_kernel_ms=ref_time,
                    t_decode_ms=0.0,
                    t_total_ms=ref_time,
                    hash_out=ref_hash,
                    hash_ref=ref_hash,
                    match=True,
                    meta=meta,
                )
                results.append(res_ref.to_json())
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
    # Vulkan options
    p.add_argument("--shader", type=Path, default=ROOT / "gpu_shaders" / "sign_flip.comp")
    p.add_argument("--spv", type=Path, default=None, help="Optional SPIR-V output path; defaults beside shader.")
    p.add_argument("--device-index", type=int, default=0)
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
    out = _timestamped_path(args.out, args.suite)
    if args.suite == "pq_roundtrip":
        bench_pq_roundtrip(args.sizes, args.sparsity, args.repeats, args.seed, out)
    elif args.suite == "kernel_dense_vs_pq":
        bench_kernel_dense_vs_pq(args.sizes, args.sparsity, args.repeats, args.seed, out)
    elif args.suite == "pq_block_sweep":
        blocks = _parse_blocks(args.blocks)
        if not blocks:
            blocks = recommend_pq_block_elems()
        bench_pq_block_sweep(args.sizes, args.sparsity, blocks, args.repeats, args.seed, out)
    elif args.suite == "kernel_dense_vulkan":
        bench_kernel_dense_vulkan(
            args.sizes,
            args.sparsity,
            args.repeats,
            args.seed,
            out,
            shader=args.shader,
            spv=args.spv,
            device_index=args.device_index,
        )
    else:
        raise ValueError(f"Unknown suite: {args.suite}")


if __name__ == "__main__":
    main()
