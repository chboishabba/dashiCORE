"""
Optional vkFFT executor with two backends:
- pyvkfft (OpenCL/CUDA) when available; no Vulkan handles needed.
- Vulkan-bound vkFFT when provided handles are passed in.
Design notes:
- Plans are cached by (shape, dtype, direction, device) and reuse buffers (Vulkan path).
- Opt-in via fft_backend="vkfft"; default is pure NumPy. No implicit handle creation.
- On missing vkFFT bindings or any binding failure, emits a single warning and falls back to NumPy.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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

try:  # Optional dependency; OpenCL/CUDA wrapper around VkFFT.
    from pyvkfft import VkFFTApp as PyVkFFTApp  # type: ignore
except Exception:  # pragma: no cover - executed only when pyvkfft is missing
    PyVkFFTApp = None  # type: ignore


def has_vkfft() -> bool:
    """Return True if any VkFFT binding (pyvkfft or vulkan vkfft) is importable."""
    return vkfft is not None or PyVkFFTApp is not None


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
            fft_backend: "numpy" (default) or "vkfft". Any other value defaults to "numpy".
        """
        self.handles = handles
        self.fft_backend = fft_backend if fft_backend in ("numpy", "vkfft") else "numpy"
        self._plans: Dict[_PlanKey, _PlanCtx] = {}
        self._command_pool = None
        self._warned = False

    # ----------------------------- public API -----------------------------
    def fft2(self, x: np.ndarray) -> np.ndarray:
        plan = self._get_plan(x, direction="fft")
        if plan is None:
            return np.fft.fft2(x)
        return self._execute(plan, x, inverse=False)

    def ifft2(self, x: np.ndarray) -> np.ndarray:
        plan = self._get_plan(x, direction="ifft")
        if plan is None:
            return np.fft.ifft2(x)
        return self._execute(plan, x, inverse=True)

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

    def _can_use_vkfft(self) -> bool:
        if self.fft_backend != "vkfft":
            return False
        if PyVkFFTApp is not None:
            return True
        if vkfft is None:
            self._warn_once("vkFFT not installed; using NumPy FFT.")
            return False
        if vk is None:
            self._warn_once("Vulkan python bindings missing; using NumPy FFT.")
            return False
        if self.handles is None:
            self._warn_once("vkFFT requested but no Vulkan handles provided; using NumPy FFT.")
            return False
        return True

    def _get_plan(self, arr: np.ndarray, direction: str) -> Optional[_PlanCtx]:
        key = _PlanKey(shape=tuple(arr.shape), dtype=arr.dtype, direction=direction, device_id=self._device_id())
        if key in self._plans:
            return self._plans[key]
        if not self._can_use_vkfft():
            return None
        if PyVkFFTApp is not None:
            plan = self._build_pyvkfft_plan(arr, direction, key)
        else:
            plan = self._build_vulkan_plan(arr, direction, key)
        if plan is None:
            return None
        self._plans[key] = plan
        return plan

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
        if PyVkFFTApp is None:
            return None
        try:
            app = PyVkFFTApp(shape=arr.shape, dtype=arr.dtype, backend="opencl")  # type: ignore[arg-type]
            return _PlanCtx(key=key, backend="pyvkfft", app=app, bytes_len=arr.nbytes)
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
        # Handle different binding styles:
        # 1) VkFFTApp instance with fft/ifft methods.
        # 2) Module-level fft/ifft callables.
        calls = []
        if hasattr(app, "ifft") and hasattr(app, "fft"):
            calls.append(lambda: app.ifft(plan.device_buffer, plan.device_buffer) if inverse else app.fft(plan.device_buffer, plan.device_buffer))  # type: ignore[misc]
        if hasattr(app, "fft"):
            calls.append(lambda: app.fft(plan.device_buffer, plan.device_buffer, inverse=inverse))  # type: ignore[misc]
        if hasattr(vkfft, "fft"):
            calls.append(lambda: vkfft.ifft(plan.device_buffer, ndim=2) if inverse else vkfft.fft(plan.device_buffer, ndim=2))  # type: ignore[misc]
        for fn in calls:
            try:
                fn()
                return
            except Exception:
                continue
        raise RuntimeError("vkFFT bindings do not expose a usable fft/ifft call")

    def _run_pyvkfft(self, plan: _PlanCtx, arr: np.ndarray, *, inverse: bool) -> np.ndarray:
        app = plan.app
        if hasattr(app, "ifft") and hasattr(app, "fft"):
            return app.ifft(arr) if inverse else app.fft(arr)  # type: ignore[misc]
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
