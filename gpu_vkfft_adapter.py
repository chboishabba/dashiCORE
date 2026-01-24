"""
Optional vkFFT executor with selectable backends:
- pyvkfft OpenCL when available; no Vulkan handles needed.
- Vulkan-bound vkFFT when provided handles are passed in.
Design notes:
- Plans are cached by (shape, dtype, direction, device) and reuse buffers (Vulkan path).
- Opt-in via fft_backend in {"vkfft", "vkfft-opencl", "vkfft-vulkan"}; default is pure NumPy.
- On missing vkFFT bindings or any binding failure, emits a single warning and falls back to NumPy.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Dict, Iterable, Optional, Tuple, Any

import numpy as np

from gpu_vulkan_dispatcher import (
    HOST_VISIBLE_COHERENT,
    VulkanHandles,
    _create_buffer,
    _read_buffer,
    _write_buffer,
)

try:  # Optional dependency; never hard-require at import time.
    import vulkan as vk  # type: ignore
except Exception:  # pragma: no cover - executed only when Vulkan is missing
    vk = None  # type: ignore

try:  # Optional dependency; present only when vkFFT Python bindings are installed.
    import vkfft  # type: ignore
except Exception:  # pragma: no cover - executed only when vkFFT is missing
    vkfft = None  # type: ignore

try:  # Optional dependency; pybind11 Vulkan bridge built from DTolm/VkFFT.
    import vkfft_vulkan_py  # type: ignore
except Exception:  # pragma: no cover - executed only when module is missing
    vkfft_vulkan_py = None  # type: ignore

try:  # Optional dependency; OpenCL wrapper around VkFFT.
    from pyvkfft.opencl import VkFFTApp as OpenCLVkFFTApp  # type: ignore
except Exception:  # pragma: no cover - executed only when pyvkfft is missing
    OpenCLVkFFTApp = None  # type: ignore


def has_vkfft() -> bool:
    """Return True if any VkFFT binding (pyvkfft or vulkan vkfft) is importable."""
    return vkfft is not None or vkfft_vulkan_py is not None or OpenCLVkFFTApp is not None


_pyvkfft_probe_ran = False
_pyvkfft_safe = False


def _probe_pyvkfft_safe(timeout: float = 5.0) -> bool:
    """
    Probe pyvkfft/OpenCL in a subprocess so driver crashes don't take down the
    main process. Returns True only if a tiny fft/ifft succeeds.
    """

    def _worker(q: Queue) -> None:
        try:
            import numpy as _np
            import pyopencl as _cl
            from pyvkfft.opencl import VkFFTApp as _App  # type: ignore

            ctx = _cl.create_some_context(interactive=False)
            queue = _cl.CommandQueue(ctx)
            x = _np.zeros((8, 8), dtype=_np.complex64)
            buf = _cl.Buffer(
                ctx, _cl.mem_flags.READ_WRITE | _cl.mem_flags.COPY_HOST_PTR, hostbuf=x
            )
            app = _App(x.shape, x.dtype, queue=queue)
            app.fft(buf)
            app.ifft(buf)
            _cl.enqueue_copy(queue, x, buf).wait()
            q.put(True)
        except Exception:
            q.put(False)

    q: Queue = Queue()
    p = Process(target=_worker, args=(q,), daemon=True)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.kill()
        return False
    if p.exitcode != 0:
        return False
    return not q.empty() and bool(q.get())


@dataclass(frozen=True)
class _PlanKey:
    shape: Tuple[int, ...]
    dtype: np.dtype
    direction: str  # "fft" or "ifft"
    device_id: Optional[int]


@dataclass
class _PlanCtx:
    key: _PlanKey
    backend: str  # "vulkan" or "pyvkfft"
    app: object
    device_buffer: Optional[object] = None
    device_memory: Optional[object] = None
    staging_buffer: Optional[object] = None
    staging_memory: Optional[object] = None
    scratch_buffer: Optional[object] = None
    scratch_memory: Optional[object] = None
    bytes_len: int = 0


class VkFFTExecutor:
    """vkFFT-backed FFT executor with plan caching and safe NumPy fallback."""

    def __init__(
        self,
        *,
        handles: Optional[VulkanHandles] = None,
        fft_backend: str = "numpy",
    ):
        """
        Args:
            handles: Shared Vulkan handles (device/queue/allocator) to bind vkFFT to.
            fft_backend: one of "numpy", "vkfft" (auto), "vkfft-opencl", "vkfft-vulkan".
        """
        self.handles = handles
        self.fft_backend = self._normalize_backend(fft_backend)
        self._plans: Dict[_PlanKey, _PlanCtx] = {}
        self._command_pool = None
        self._warned = False

    def _normalize_backend(self, backend: str) -> str:
        allowed = {"numpy", "vkfft", "vkfft-opencl", "vkfft-vulkan"}
        return backend if backend in allowed else "numpy"

    # ----------------------------- public API -----------------------------
    def fft2(self, x: np.ndarray) -> np.ndarray:
        arr = self._coerce_input(x)
        plan = self._get_plan(arr, direction="fft")
        if plan is None:
            return np.fft.fft2(arr)
        return self._execute(plan, arr, inverse=False)

    def ifft2(self, x: np.ndarray) -> np.ndarray:
        arr = self._coerce_input(x)
        plan = self._get_plan(arr, direction="ifft")
        if plan is None:
            return np.fft.ifft2(arr)
        return self._execute(plan, arr, inverse=True)

    def close(self) -> None:
        """Release any GPU resources owned by this executor (plans, pools)."""
        if vk is not None:
            for ctx in self._plans.values():
                self._destroy_plan_buffers(ctx)
            if self._command_pool is not None:
                vk.vkDestroyCommandPool(self.handles.device, self._command_pool, None)  # type: ignore[arg-type]
                self._command_pool = None
        self._plans.clear()

    # ----------------------------- internal helpers -----------------------------
    def _warn_once(self, msg: str) -> None:
        if not self._warned:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            self._warned = True

    def _coerce_input(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x)
        if self.fft_backend == "numpy":
            return arr
        # Vulkan binding assumes complex64; promote real or higher-precision complex to complex64.
        if not np.iscomplexobj(arr) or arr.dtype not in (np.complex64,):
            return arr.astype(np.complex64, copy=True)
        return arr

    def _as_u64(self, obj: Any) -> int:
        """Best-effort conversion of Vulkan cffi handles to integer addresses."""
        if obj is None:
            raise ValueError("handle is None")
        try:
            return int(obj)
        except Exception:
            pass
        for attr in ("handle", "value"):
            if hasattr(obj, attr):
                try:
                    return int(getattr(obj, attr))
                except Exception:
                    continue
        # cffi pointer from python-vulkan
        if vk is not None and hasattr(vk, "ffi"):
            try:
                return int(vk.ffi.cast("uintptr_t", obj))
            except Exception:
                pass
        raise TypeError("Expected an int-like Vulkan handle (or object with .handle/.value)")

    def _can_use_vkfft(self) -> bool:
        if self.fft_backend == "numpy":
            return False

        # Auto mode prefers Vulkan when handles + binding exist, else OpenCL via pyvkfft.
        if self.fft_backend == "vkfft":
            if vkfft_vulkan_py is not None and vk is not None and self.handles is not None:
                return True
            if vkfft is not None and vk is not None and self.handles is not None:
                return True
            if OpenCLVkFFTApp is not None:
                return True
            self._warn_once("vkFFT not installed; using NumPy FFT.")
            return False

        if self.fft_backend == "vkfft-vulkan":
            if vkfft_vulkan_py is None and vkfft is None:
                self._warn_once("Vulkan vkFFT binding missing; build vkfft_vulkan_py via setup_vkfft_vulkan.py.")
                return False
            if vk is None:
                self._warn_once("Vulkan python bindings missing; using NumPy FFT.")
                return False
            if self.handles is None:
                self._warn_once("vkFFT/Vulkan requested but no Vulkan handles provided; using NumPy FFT.")
                return False
            return True

        if self.fft_backend == "vkfft-opencl":
            if OpenCLVkFFTApp is None:
                self._warn_once("pyvkfft (OpenCL) not installed; using NumPy FFT.")
                return False
            return True

        return False

    def _get_plan(self, arr: np.ndarray, direction: str) -> Optional[_PlanCtx]:
        key = _PlanKey(shape=tuple(arr.shape), dtype=arr.dtype, direction=direction, device_id=self._device_id())
        if key in self._plans:
            return self._plans[key]
        if not self._can_use_vkfft():
            return None

        plan: Optional[_PlanCtx] = None
        # Ordering: explicit backend choice; auto prefers Vulkan then OpenCL.
        backend_order = self._backend_order()
        for backend in backend_order:
            if backend == "vulkan":
                plan = self._build_vulkan_plan(arr, direction, key)
            elif backend == "pyvkfft":
                plan = self._build_pyvkfft_plan(arr, direction, key)
            if plan is not None:
                break
        if plan is None:
            return None
        self._plans[key] = plan
        return plan

    def _backend_order(self) -> Iterable[str]:
        if self.fft_backend == "vkfft-vulkan":
            return ("vulkan",)
        if self.fft_backend == "vkfft-opencl":
            return ("pyvkfft",)
        # auto: prefer Vulkan when handles/binding exist, else OpenCL
        return ("vulkan", "pyvkfft")

    def _build_vulkan_plan(self, arr: np.ndarray, direction: str, key: _PlanKey) -> Optional[_PlanCtx]:
        assert self.handles is not None
        try:
            bytes_len = arr.nbytes
            usage = (
                vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
            )

            # Device-local working buffer
            device_buffer, device_memory = self.handles.create_buffer(
                bytes_len, usage, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            )
            # Scratch buffer (device-local) reused per plan
            scratch_buffer, scratch_memory = self.handles.create_buffer(
                bytes_len, usage, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            )
            # Host-visible staging for IO
            staging_buffer, staging_memory = _create_buffer(
                self.handles.device,
                self.handles.mem_props,
                bytes_len,
                vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                HOST_VISIBLE_COHERENT,
            )

            if vkfft_vulkan_py is not None:
                buffer_size = plan_bytes = bytes_len
                scratch_size = bytes_len
                app = vkfft_vulkan_py.VkFFTPlan(
                    physical_device=self._as_u64(self.handles.physical_device),
                    device=self._as_u64(self.handles.device),
                    queue=self._as_u64(self.handles.queue),
                    command_pool=self._as_u64(self._ensure_command_pool()),
                    queue_family_index=int(self.handles.queue_family_index),
                    buffer_size_bytes=buffer_size,
                    scratch_size_bytes=scratch_size,
                    dims=list(arr.shape),
                    buffer=self._as_u64(device_buffer),
                    scratch_buffer=self._as_u64(scratch_buffer),
                    inverse=direction == "ifft",
                )
            else:
                cfg = {
                    "device": self.handles.device,
                    "physical_device": self.handles.physical_device,
                    "queue": self.handles.queue,
                    "queue_family_index": self.handles.queue_family_index,
                    "command_pool": self._ensure_command_pool(),
                    "allocator": self.handles.allocator,
                    "buffer": device_buffer,
                    "scratch_buffer": scratch_buffer,
                }
                app = self._create_vkfft_app(arr.shape, arr.dtype, cfg)
            return _PlanCtx(
                key=key,
                backend="vulkan",
                app=app,
                device_buffer=device_buffer,
                device_memory=device_memory,
                staging_buffer=staging_buffer,
                staging_memory=staging_memory,
                scratch_buffer=scratch_buffer,
                scratch_memory=scratch_memory,
                bytes_len=bytes_len,
            )
        except Exception as exc:  # pragma: no cover - exercised only when vkFFT bindings misbehave
            self._warn_once(f"vkFFT binding failed ({exc}); using NumPy FFT.")
            return None

    def _build_pyvkfft_plan(self, arr: np.ndarray, direction: str, key: _PlanKey) -> Optional[_PlanCtx]:
        if OpenCLVkFFTApp is None:
            return None
        global _pyvkfft_probe_ran, _pyvkfft_safe
        if not _pyvkfft_probe_ran:
            _pyvkfft_safe = _probe_pyvkfft_safe()
            _pyvkfft_probe_ran = True
            if not _pyvkfft_safe:
                self._warn_once("pyvkfft (OpenCL) probed and crashed; using NumPy FFT.")
                return None
        try:
            import pyopencl as cl  # local import to avoid hard dependency
            import pyopencl.array as cla

            # Pick first available device; extend later with selection hooks.
            platforms = cl.get_platforms()
            devices = [d for p in platforms for d in p.get_devices()]
            if not devices:
                raise RuntimeError("no OpenCL devices available")
            ctx = cl.Context([devices[0]])
            queue = cl.CommandQueue(ctx)

            cl_arr = cla.to_device(queue, arr)
            app = OpenCLVkFFTApp(shape=arr.shape, dtype=arr.dtype, queue=queue)  # type: ignore[arg-type]
            return _PlanCtx(
                key=key,
                backend="pyvkfft",
                app=app,
                device_buffer=cl_arr,
                bytes_len=arr.nbytes,
            )
        except Exception as exc:  # pragma: no cover - exercised only when pyvkfft misbehaves
            self._warn_once(f"pyvkfft binding failed ({exc}); using NumPy FFT.")
            return None

    def _create_vkfft_app(self, shape: Tuple[int, ...], dtype: np.dtype, cfg: dict):
        """Isolated to keep the try/except tight around binding differences."""
        # pyvkfft expects a tuple shape and dtype; adapter classes vary slightly across builds.
        if hasattr(vkfft, "VkFFTApp"):
            return vkfft.VkFFTApp(shape=shape, dtype=dtype, backend="vulkan", config=cfg)  # type: ignore[arg-type]
        # Fallback: older bindings expose fft/ifft callables directly.
        return vkfft

    def _execute(self, plan: _PlanCtx, arr: np.ndarray, *, inverse: bool) -> np.ndarray:
        try:
            if plan.backend == "pyvkfft":
                return self._run_pyvkfft(plan, arr, inverse=inverse)
            self._upload(arr, plan)
            self._run_vkfft(plan, inverse=inverse)
            return self._download(plan, arr.shape, arr.dtype)
        except Exception as exc:  # pragma: no cover - fallback path
            self._warn_once(f"vkFFT execution failed ({exc}); using NumPy FFT.")
            if inverse:
                return np.fft.ifft2(arr)
            return np.fft.fft2(arr)

    def _upload(self, arr: np.ndarray, plan: _PlanCtx) -> None:
        if plan.backend != "vulkan":
            return
        _write_buffer(self.handles.device, plan.staging_memory, arr)  # type: ignore[arg-type]
        cmd = self._allocate_command_buffer()
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(cmd, begin_info)
        vk.vkCmdCopyBuffer(
            cmd,
            plan.staging_buffer,
            plan.device_buffer,
            1,
            [vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=plan.bytes_len)],
        )
        barrier = vk.VkBufferMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            buffer=plan.device_buffer,
            offset=0,
            size=plan.bytes_len,
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            None,
            1,
            [barrier],
            0,
            None,
        )
        vk.vkEndCommandBuffer(cmd)
        self._submit_and_wait(cmd)

    def _run_vkfft(self, plan: _PlanCtx, *, inverse: bool) -> None:
        app = plan.app
        if hasattr(app, "exec"):
            app.exec()
            return
        # Handle different binding styles:
        # 1) VkFFTApp instance with fft/ifft methods.
        # 2) Module-level fft/ifft callables.
        calls = []
        if hasattr(app, "ifft") and hasattr(app, "fft"):
            calls.append(lambda: app.ifft(plan.device_buffer, plan.device_buffer) if inverse else app.fft(plan.device_buffer, plan.device_buffer))  # type: ignore[misc]
        if hasattr(app, "fft"):
            calls.append(lambda: app.fft(plan.device_buffer, plan.device_buffer, inverse=inverse))  # type: ignore[misc]
        if vkfft is not None and hasattr(vkfft, "fft"):
            calls.append(lambda: vkfft.ifft(plan.device_buffer, ndim=2) if inverse else vkfft.fft(plan.device_buffer, ndim=2))  # type: ignore[misc]
        for fn in calls:
            try:
                fn()
                return
            except Exception:
                continue
        raise RuntimeError("vkFFT bindings do not expose a usable fft/ifft call")

    def _run_pyvkfft(self, plan: _PlanCtx, arr: np.ndarray, *, inverse: bool) -> np.ndarray:
        import pyopencl.array as cla

        # Reuse the preallocated device buffer when shapes match; else recreate.
        cl_arr = plan.device_buffer
        if cl_arr is None or tuple(cl_arr.shape) != tuple(arr.shape):  # type: ignore[attr-defined]
            cl_arr = cla.to_device(plan.app.queue, arr)
            plan.device_buffer = cl_arr
        else:
            cl_arr.set(arr)

        app = plan.app
        if hasattr(app, "ifft") and hasattr(app, "fft"):
            if inverse:
                app.ifft(cl_arr)
            else:
                app.fft(cl_arr)
            return cl_arr.get()
        raise RuntimeError("pyvkfft VkFFTApp missing fft/ifft")

    def _download(self, plan: _PlanCtx, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        if plan.backend != "vulkan":
            raise RuntimeError("download called for non-Vulkan plan")
        cmd = self._allocate_command_buffer()
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(cmd, begin_info)
        barrier = vk.VkBufferMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            buffer=plan.device_buffer,
            offset=0,
            size=plan.bytes_len,
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            None,
            1,
            [barrier],
            0,
            None,
        )
        vk.vkCmdCopyBuffer(
            cmd,
            plan.device_buffer,
            plan.staging_buffer,
            1,
            [vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=plan.bytes_len)],
        )
        vk.vkEndCommandBuffer(cmd)
        self._submit_and_wait(cmd)
        return _read_buffer(self.handles.device, plan.staging_memory, shape, dtype)  # type: ignore[arg-type]

    def _allocate_command_buffer(self):
        if self.handles is None:
            raise RuntimeError("Vulkan handles required for command buffer allocation")
        pool = self._ensure_command_pool()
        alloc_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        return vk.vkAllocateCommandBuffers(self.handles.device, alloc_info)[0]  # type: ignore[arg-type]

    def _ensure_command_pool(self):
        if self.handles is None:
            raise RuntimeError("Vulkan handles required for command pool")
        if self._command_pool is not None:
            return self._command_pool
        pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=self.handles.queue_family_index,  # type: ignore[union-attr]
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        )
        self._command_pool = vk.vkCreateCommandPool(self.handles.device, pool_info, None)  # type: ignore[arg-type]
        return self._command_pool

    def _submit_and_wait(self, cmd):
        if self.handles is None:
            raise RuntimeError("Vulkan handles required for submit")
        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[cmd],
        )
        fence_info = vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
        fence = vk.vkCreateFence(self.handles.device, fence_info, None)  # type: ignore[arg-type]
        vk.vkQueueSubmit(self.handles.queue, 1, [submit_info], fence)  # type: ignore[arg-type]
        vk.vkWaitForFences(self.handles.device, 1, [fence], vk.VK_TRUE, 0xFFFFFFFFFFFFFFFF)  # type: ignore[arg-type]
        vk.vkDestroyFence(self.handles.device, fence, None)  # type: ignore[arg-type]

    def _destroy_plan_buffers(self, ctx: _PlanCtx) -> None:
        if ctx.backend != "vulkan":
            return
        vk.vkDestroyBuffer(self.handles.device, ctx.device_buffer, None)  # type: ignore[arg-type]
        vk.vkFreeMemory(self.handles.device, ctx.device_memory, None)  # type: ignore[arg-type]
        vk.vkDestroyBuffer(self.handles.device, ctx.scratch_buffer, None)  # type: ignore[arg-type]
        vk.vkFreeMemory(self.handles.device, ctx.scratch_memory, None)  # type: ignore[arg-type]
        vk.vkDestroyBuffer(self.handles.device, ctx.staging_buffer, None)  # type: ignore[arg-type]
        vk.vkFreeMemory(self.handles.device, ctx.staging_memory, None)  # type: ignore[arg-type]

    def _device_id(self) -> Optional[int]:
        if self.handles is None:
            return None
        try:
            return int(self.handles.device)  # type: ignore[arg-type]
        except Exception:
            return id(self.handles.device)


__all__ = ["VkFFTExecutor", "has_vkfft"]
