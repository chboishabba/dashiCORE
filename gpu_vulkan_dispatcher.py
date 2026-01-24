from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from dashi_core.carrier import Carrier
from gpu_common_methods import compile_shader, find_memory_type
from gpu_vulkan_adapter import DispatchFn, VulkanKernelConfig

try:
    import vulkan as vk
except ImportError:  # pragma: no cover - exercised only when Vulkan is missing
    vk = None  # type: ignore[assignment]


HOST_VISIBLE_COHERENT = (
    (1 << 1)  # VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    | (1 << 2)  # VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
)


def _now_ms() -> float:
    return time.perf_counter_ns() / 1e6


@dataclass(frozen=True)
class VulkanDispatchConfig:
    """Device/queue selection for Vulkan carrier dispatch."""

    device_index: int = 0
    queue_family_index: Optional[int] = None
    memory_mode: str = "host_visible"  # "host_visible" (zero-copy) or "device_local" (staged, VRAM-resident)


@dataclass(frozen=True)
class DispatchTiming:
    """Capture timing for a single submit→fence + host wall."""

    submit_to_fence_ms: float
    wall_ms: float
    fence_waits: int = 1


def _require_vk() -> None:
    if vk is None:
        raise RuntimeError("vulkan python package not installed; install via `pip install vulkan`")


@dataclass
class VulkanHandles:
    """Bundle of Vulkan objects that can be shared across dispatchers/backends."""

    instance: object
    physical_device: object
    device: object
    queue: object
    queue_family_index: int
    mem_props: object
    allocator: Optional[object] = None
    owns_device: bool = True
    owns_instance: bool = True

    def create_buffer(self, size: int, usage: int, required_flags: int):
        return _create_buffer(self.device, self.mem_props, size, usage, required_flags)

    def destroy_buffer(self, buffer, memory) -> None:
        vk.vkDestroyBuffer(self.device, buffer, None)
        vk.vkFreeMemory(self.device, memory, None)

    def close(self) -> None:
        if self.device and self.owns_device:
            vk.vkDestroyDevice(self.device, None)
        if self.instance and self.owns_instance:
            vk.vkDestroyInstance(self.instance, None)


def create_vulkan_handles(dispatch_config: Optional[VulkanDispatchConfig] = None) -> VulkanHandles:
    """Create and return shared Vulkan handles according to dispatch config."""
    _require_vk()
    cfg = dispatch_config or VulkanDispatchConfig()

    app_info = vk.VkApplicationInfo(
        sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName="dashiCORE",
        applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        pEngineName="dashiCORE",
        engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        apiVersion=vk.VK_MAKE_VERSION(1, 2, 0),
    )
    create_info = vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=app_info,
    )
    instance = vk.vkCreateInstance(create_info, None)

    physical_devices = vk.vkEnumeratePhysicalDevices(instance)
    if not physical_devices:
        raise RuntimeError("No Vulkan physical devices found")
    physical_device = physical_devices[cfg.device_index]
    queue_family_index = _pick_compute_queue_family(physical_device, cfg.queue_family_index)
    device, queue = _create_device_and_queue(physical_device, queue_family_index)
    mem_props = vk.vkGetPhysicalDeviceMemoryProperties(physical_device)
    return VulkanHandles(
        instance=instance,
        physical_device=physical_device,
        device=device,
        queue=queue,
        queue_family_index=queue_family_index,
        mem_props=mem_props,
    )


def _pick_compute_queue_family(physical_device, preferred: Optional[int] = None) -> int:
    queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
    if preferred is not None:
        props = queue_families[preferred]
        if props.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
            return preferred
    for idx, props in enumerate(queue_families):
        if props.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
            return idx
    raise RuntimeError("No compute-capable queue family found")


def _create_buffer(device, mem_props, size: int, usage: int, required_flags: int):
    buffer_info = vk.VkBufferCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        size=size,
        usage=usage,
        sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
    )
    buffer = vk.vkCreateBuffer(device, buffer_info, None)
    mem_req = vk.vkGetBufferMemoryRequirements(device, buffer)
    mem_type_index = find_memory_type(mem_props, mem_req.memoryTypeBits, required_flags)
    alloc_info = vk.VkMemoryAllocateInfo(
        sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=mem_req.size,
        memoryTypeIndex=mem_type_index,
    )
    memory = vk.vkAllocateMemory(device, alloc_info, None)
    vk.vkBindBufferMemory(device, buffer, memory, 0)
    return buffer, memory


def _write_buffer(device, memory, data: np.ndarray) -> None:
    mapped = vk.vkMapMemory(device, memory, 0, data.nbytes, 0)
    try:
        data_bytes = data.tobytes()
        if _try_write_via_buffer(mapped, data_bytes):
            return
        addr = _ptr_to_int(mapped)
        ctypes.memmove(ctypes.c_void_p(addr), data.ctypes.data, data.nbytes)
    finally:
        vk.vkUnmapMemory(device, memory)


def _read_buffer(device, memory, shape: Sequence[int], dtype: np.dtype) -> np.ndarray:
    bytes_len = int(np.prod(shape)) * np.dtype(dtype).itemsize
    mapped = vk.vkMapMemory(device, memory, 0, bytes_len, 0)
    try:
        try:
            mv = memoryview(mapped)
            return np.frombuffer(mv, dtype=dtype, count=int(np.prod(shape))).reshape(shape).copy()
        except TypeError:
            addr = _ptr_to_int(mapped)
            buf = ctypes.string_at(addr, bytes_len)
            return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()
    finally:
        vk.vkUnmapMemory(device, memory)


def _try_write_via_buffer(mapped, data_bytes: bytes) -> bool:
    try:
        mv = memoryview(mapped)
    except TypeError:
        return False
    try:
        mv[: len(data_bytes)] = data_bytes
        return True
    except (TypeError, ValueError):
        return False


def _ptr_to_int(ptr) -> int:
    try:
        return int(ptr)
    except (TypeError, ValueError):
        if isinstance(ptr, bytes):
            return int.from_bytes(ptr, "little")
        try:
            return ctypes.addressof(ctypes.c_char.from_buffer(ptr))
        except TypeError:
            raise TypeError("Cannot convert mapped pointer to address")


def _create_device_and_queue(physical_device, queue_family_index: int):
    priorities = [1.0]
    queue_info = vk.VkDeviceQueueCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        queueFamilyIndex=queue_family_index,
        queueCount=1,
        pQueuePriorities=priorities,
    )
    device_info = vk.VkDeviceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        queueCreateInfoCount=1,
        pQueueCreateInfos=[queue_info],
        pEnabledFeatures=vk.VkPhysicalDeviceFeatures(),
    )
    device = vk.vkCreateDevice(physical_device, device_info, None)
    queue = vk.vkGetDeviceQueue(device, queue_family_index, 0)
    return device, queue


class VulkanCarrierDispatcher:
    """
    Minimal Vulkan compute dispatcher for Carrier tensors.

    Uses a storage-buffer GLSL compute shader compiled to SPIR-V. Expects four bindings:
    sign_in (int32), support_in (uint32), sign_out (int32), support_out (uint32).
    """

    def __init__(
        self,
        config: VulkanKernelConfig,
        dispatch_config: Optional[VulkanDispatchConfig] = None,
        *,
        shared_handles: Optional[VulkanHandles] = None,
    ):
        _require_vk()
        self.config = config
        self.dispatch_config = dispatch_config or VulkanDispatchConfig()
        self.shared_handles = shared_handles

    def __call__(self, carrier: Carrier) -> Carrier:
        return self.dispatch(carrier)

    def dispatch(self, carrier: Carrier) -> Carrier:
        """Single-dispatch convenience wrapper (kept for adapter compatibility)."""
        out, _, _ = self.dispatch_batched(carrier, dispatches=1, collect_timing=False)
        return out

    def dispatch_batched(
        self,
        carrier: Carrier,
        dispatches: int = 1,
        *,
        collect_timing: bool = False,
        hash_only: bool = False,
        handles: Optional[VulkanHandles] = None,
    ):
        """Record `dispatches` dispatches into one command buffer, submit once, wait once.

        When `hash_only` is True (device_local path), compute a GPU-side hash of outputs and
        avoid full readback; returns (Carrier | None, DispatchTiming | None, hash_value | None).
        """
        _require_vk()
        compile_shader(self.config.shader_path, self.config.spv_path)

        active_handles: Optional[VulkanHandles] = handles or self.shared_handles
        owns_handles = active_handles is None
        if active_handles is None:
            active_handles = create_vulkan_handles(self.dispatch_config)

        device = active_handles.device
        queue = active_handles.queue
        mem_props = active_handles.mem_props
        instance = active_handles.instance
        physical_device = active_handles.physical_device
        queue_family_index = active_handles.queue_family_index

        sign_in, support_in = self._prepare_inputs(carrier)
        sign_out = np.empty_like(sign_in)
        support_out = np.empty_like(support_in)

        buffers = []
        memories = []
        staging_buffers = []
        staging_memories = []
        hash_buffer = None
        hash_memory = None
        hash_staging_buffer = None
        hash_staging_memory = None
        memory_mode = getattr(self.dispatch_config, "memory_mode", "host_visible")
        if memory_mode not in ("host_visible", "device_local"):
            memory_mode = "host_visible"
        submit_ms = None
        wall_start = _now_ms()
        try:
            if memory_mode == "device_local":
                # Device-local buffers for compute, host-visible staging for IO
                usage_storage = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                usage_copy_src = vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                usage_copy_dst = vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
                hash_shader_path = self.config.shader_path.parent / "hash_reduce.comp"
                hash_spv_path = hash_shader_path.with_suffix(".spv")
                if hash_only:
                    compile_shader(hash_shader_path, hash_spv_path)

                # Staging inputs (host visible, copy src)
                for arr in (sign_in, support_in):
                    buf, mem = _create_buffer(device, mem_props, arr.nbytes, usage_copy_src, HOST_VISIBLE_COHERENT)
                    staging_buffers.append(buf)
                    staging_memories.append(mem)
                # Device buffers (storage + copy src/dst)
                for arr in (sign_in, support_in, sign_out, support_out):
                    buf, mem = _create_buffer(
                        device,
                        mem_props,
                        arr.nbytes,
                        usage_storage | usage_copy_src | usage_copy_dst,
                        vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    )
                    buffers.append(buf)
                    memories.append(mem)
                # Device-local hash buffer (4 bytes) plus staging hash buffer
                hash_buffer, hash_memory = _create_buffer(
                    device,
                    mem_props,
                    4,
                    usage_storage | usage_copy_src | usage_copy_dst,
                    vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                )
                hash_staging_buffer, hash_staging_memory = _create_buffer(
                    device,
                    mem_props,
                    4,
                    usage_copy_dst,
                    HOST_VISIBLE_COHERENT,
                )
                # Staging outputs (host visible, copy dst)
                for arr in (sign_out, support_out):
                    buf, mem = _create_buffer(device, mem_props, arr.nbytes, usage_copy_dst, HOST_VISIBLE_COHERENT)
                    staging_buffers.append(buf)
                    staging_memories.append(mem)

                # Upload inputs to staging
                _write_buffer(device, staging_memories[0], sign_in)
                _write_buffer(device, staging_memories[1], support_in)

                descriptor_set_layout, pipeline_layout = self._build_layouts(device)
                shader_module = self._create_shader_module(device, self.config.spv_path)
                pipeline = self._create_pipeline(device, pipeline_layout, shader_module)
                descriptor_pool, descriptor_set = self._allocate_descriptors(device, descriptor_set_layout)
                self._bind_buffers(device, descriptor_set, buffers, [sign_in, support_in, sign_out, support_out])
                command_pool, command_buffer = self._build_command_buffer(device, queue_family_index)

                hash_handles: dict = {}
                submit_ms = self._record_with_copies(
                    device=device,
                    command_buffer=command_buffer,
                    pipeline=pipeline,
                    pipeline_layout=pipeline_layout,
                    descriptor_set=descriptor_set,
                    queue=queue,
                    global_size=sign_in.shape[0],
                    staging_in=staging_buffers[:2],
                    device_in=buffers[:2],
                    staging_out=staging_buffers[2:],
                    device_out=buffers[2:],
                    sign_nbytes=sign_in.nbytes,
                    support_nbytes=support_in.nbytes,
                    dispatch_count=max(1, int(dispatches)),
                    hash_only=hash_only,
                    hash_buffer=hash_buffer,
                    hash_staging_buffer=hash_staging_buffer,
                    hash_push_constant=sign_in.shape[0],
                    hash_spv_path=hash_spv_path if hash_only else None,
                    hash_handles=hash_handles,
                )

                if not hash_only:
                    sign_result = _read_buffer(device, staging_memories[2], sign_out.shape, sign_out.dtype)
                    support_result = _read_buffer(device, staging_memories[3], support_out.shape, support_out.dtype)
                else:
                    sign_result = None
                    support_result = None
                hash_value = None
                if hash_only:
                    mapped = vk.vkMapMemory(device, hash_staging_memory, 0, 4, 0)
                    try:
                        try:
                            mv = memoryview(mapped)
                            hash_value = int(np.frombuffer(mv, dtype=np.uint32, count=1)[0])
                        except TypeError:
                            addr = _ptr_to_int(mapped)
                            buf = ctypes.string_at(addr, 4)
                            hash_value = int(np.frombuffer(buf, dtype=np.uint32, count=1)[0])
                    finally:
                        vk.vkUnmapMemory(device, hash_staging_memory)
            else:
                host_flags = HOST_VISIBLE_COHERENT
                buffer_usage = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

                for arr in (sign_in, support_in, sign_out, support_out):
                    buffer, memory = _create_buffer(device, mem_props, arr.nbytes, buffer_usage, host_flags)
                    buffers.append(buffer)
                    memories.append(memory)

                _write_buffer(device, memories[0], sign_in)
                _write_buffer(device, memories[1], support_in)

                descriptor_set_layout, pipeline_layout = self._build_layouts(device)
                shader_module = self._create_shader_module(device, self.config.spv_path)
                pipeline = self._create_pipeline(device, pipeline_layout, shader_module)
                descriptor_pool, descriptor_set = self._allocate_descriptors(device, descriptor_set_layout)
                self._bind_buffers(device, descriptor_set, buffers, [sign_in, support_in, sign_out, support_out])
                command_pool, command_buffer = self._build_command_buffer(device, queue_family_index)

                submit_ms = self._record_and_submit(
                    device=device,
                    command_buffer=command_buffer,
                    pipeline=pipeline,
                    pipeline_layout=pipeline_layout,
                    descriptor_set=descriptor_set,
                    queue=queue,
                    global_size=sign_in.shape[0],
                    dispatch_count=max(1, int(dispatches)),
                )

                sign_result = _read_buffer(device, memories[2], sign_out.shape, sign_out.dtype)
                support_result = _read_buffer(device, memories[3], support_out.shape, support_out.dtype)
                hash_value = None

            carrier_out = None
            if sign_result is not None and support_result is not None:
                support = support_result.astype(bool)
                sign = np.asarray(np.sign(sign_result), dtype=np.int8)
                carrier_out = Carrier(support=support, sign=sign)
            timing = None
            if collect_timing and submit_ms is not None:
                wall_ms = _now_ms() - wall_start
                timing = DispatchTiming(submit_to_fence_ms=submit_ms, wall_ms=wall_ms, fence_waits=1)
            return carrier_out, timing, hash_value
        finally:
            # Clean up resources (reverse order of creation where relevant)
            if 'command_pool' in locals():
                vk.vkDestroyCommandPool(device, command_pool, None)
            if 'descriptor_pool' in locals():
                vk.vkDestroyDescriptorPool(device, descriptor_pool, None)
            if 'pipeline' in locals():
                vk.vkDestroyPipeline(device, pipeline, None)
            if 'pipeline_layout' in locals():
                vk.vkDestroyPipelineLayout(device, pipeline_layout, None)
            if 'descriptor_set_layout' in locals():
                vk.vkDestroyDescriptorSetLayout(device, descriptor_set_layout, None)
            if 'shader_module' in locals():
                vk.vkDestroyShaderModule(device, shader_module, None)
            for buf, mem in zip(staging_buffers[::-1], staging_memories[::-1]):
                vk.vkDestroyBuffer(device, buf, None)
                vk.vkFreeMemory(device, mem, None)
            for buf, mem in zip(buffers[::-1], memories[::-1]):
                vk.vkDestroyBuffer(device, buf, None)
                vk.vkFreeMemory(device, mem, None)
            if hash_buffer is not None:
                vk.vkDestroyBuffer(device, hash_buffer, None)
            if hash_memory is not None:
                vk.vkFreeMemory(device, hash_memory, None)
            if hash_staging_buffer is not None:
                vk.vkDestroyBuffer(device, hash_staging_buffer, None)
            if hash_staging_memory is not None:
                vk.vkFreeMemory(device, hash_staging_memory, None)
            if 'hash_handles' in locals():
                h = hash_handles
                if h.get("hash_descriptor_pool"):
                    vk.vkDestroyDescriptorPool(device, h["hash_descriptor_pool"], None)
                if h.get("hash_pipeline"):
                    vk.vkDestroyPipeline(device, h["hash_pipeline"], None)
                if h.get("hash_pipeline_layout"):
                    vk.vkDestroyPipelineLayout(device, h["hash_pipeline_layout"], None)
                if h.get("hash_set_layout"):
                    vk.vkDestroyDescriptorSetLayout(device, h["hash_set_layout"], None)
                if h.get("hash_shader"):
                    vk.vkDestroyShaderModule(device, h["hash_shader"], None)
            if owns_handles and active_handles is not None:
                active_handles.close()

    def _prepare_inputs(self, carrier: Carrier) -> Tuple[np.ndarray, np.ndarray]:
        sign_in = carrier.sign.astype(np.int32, copy=False)
        support_in = carrier.support.astype(np.uint32, copy=False)
        flat_sign = sign_in.reshape(-1)
        flat_support = support_in.reshape(-1)
        return flat_sign, flat_support

    def _build_layouts(self, device):
        bindings = []
        for binding in range(4):
            bindings.append(
                vk.VkDescriptorSetLayoutBinding(
                    binding=binding,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                    pImmutableSamplers=None,
                )
            )
        layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings,
        )
        descriptor_set_layout = vk.vkCreateDescriptorSetLayout(device, layout_info, None)
        pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[descriptor_set_layout],
        )
        pipeline_layout = vk.vkCreatePipelineLayout(device, pipeline_layout_info, None)
        return descriptor_set_layout, pipeline_layout

    def _create_shader_module(self, device, spv_path: Path):
        code_bytes = spv_path.read_bytes()
        create_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(code_bytes),
            pCode=code_bytes,
        )
        return vk.vkCreateShaderModule(device, create_info, None)

    def _build_hash_layout(self, device):
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=0,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                pImmutableSamplers=None,
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                pImmutableSamplers=None,
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=2,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                pImmutableSamplers=None,
            ),
        ]
        layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings,
        )
        descriptor_set_layout = vk.vkCreateDescriptorSetLayout(device, layout_info, None)
        push_range = vk.VkPushConstantRange(stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=4)
        pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[descriptor_set_layout],
            pushConstantRangeCount=1,
            pPushConstantRanges=[push_range],
        )
        pipeline_layout = vk.vkCreatePipelineLayout(device, pipeline_layout_info, None)
        return descriptor_set_layout, pipeline_layout

    def _create_hash_pipeline(self, device, pipeline_layout, shader_module):
        stage_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=shader_module,
            pName="main",
        )
        pipeline_info = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage_info,
            layout=pipeline_layout,
        )
        return vk.vkCreateComputePipelines(device, vk.VK_NULL_HANDLE, 1, [pipeline_info], None)[0]

    def _allocate_hash_descriptors(self, device, descriptor_set_layout):
        pool_sizes = [
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=3,
            )
        ]
        pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
            maxSets=1,
        )
        descriptor_pool = vk.vkCreateDescriptorPool(device, pool_info, None)
        alloc_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[descriptor_set_layout],
        )
        descriptor_set = vk.vkAllocateDescriptorSets(device, alloc_info)[0]
        return descriptor_pool, descriptor_set

    def _bind_hash_buffers(self, device, descriptor_set, sign_buffer, support_buffer, hash_buffer):
        infos = [
            vk.VkDescriptorBufferInfo(buffer=sign_buffer, offset=0, range=vk.VK_WHOLE_SIZE),
            vk.VkDescriptorBufferInfo(buffer=support_buffer, offset=0, range=vk.VK_WHOLE_SIZE),
            vk.VkDescriptorBufferInfo(buffer=hash_buffer, offset=0, range=vk.VK_WHOLE_SIZE),
        ]
        writes = []
        for idx, info in enumerate(infos):
            writes.append(
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set,
                    dstBinding=idx,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[info],
                )
            )
        vk.vkUpdateDescriptorSets(device, len(writes), writes, 0, None)

    def _create_pipeline(self, device, pipeline_layout, shader_module):
        stage_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=shader_module,
            pName="main",
        )
        pipeline_info = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage_info,
            layout=pipeline_layout,
        )
        return vk.vkCreateComputePipelines(device, vk.VK_NULL_HANDLE, 1, [pipeline_info], None)[0]

    def _allocate_descriptors(self, device, descriptor_set_layout):
        pool_sizes = [
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=4,
            )
        ]
        pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
            maxSets=1,
        )
        descriptor_pool = vk.vkCreateDescriptorPool(device, pool_info, None)
        alloc_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[descriptor_set_layout],
        )
        descriptor_set = vk.vkAllocateDescriptorSets(device, alloc_info)[0]
        return descriptor_pool, descriptor_set

    def _bind_buffers(self, device, descriptor_set, buffers, arrays):
        writes = []
        for binding, (buffer, arr) in enumerate(zip(buffers, arrays)):
            buffer_info = vk.VkDescriptorBufferInfo(
                buffer=buffer,
                offset=0,
                range=arr.nbytes,
            )
            writes.append(
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set,
                    dstBinding=binding,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[buffer_info],
                )
            )
        vk.vkUpdateDescriptorSets(device, len(writes), writes, 0, None)

    def _build_command_buffer(self, device, queue_family_index: int):
        pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=queue_family_index,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        )
        command_pool = vk.vkCreateCommandPool(device, pool_info, None)
        alloc_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        command_buffer = vk.vkAllocateCommandBuffers(device, alloc_info)[0]
        return command_pool, command_buffer

    def _record_and_submit(
        self,
        device,
        command_buffer,
        pipeline,
        pipeline_layout,
        descriptor_set,
        queue,
        global_size: int,
        dispatch_count: int = 1,
    ) -> float:
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(command_buffer, begin_info)
        vk.vkCmdBindPipeline(command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
        vk.vkCmdBindDescriptorSets(
            command_buffer,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout,
            0,
            1,
            [descriptor_set],
            0,
            None,
        )
        gx = (global_size + self.config.workgroup[0] - 1) // self.config.workgroup[0]
        for _ in range(dispatch_count):
            vk.vkCmdDispatch(command_buffer, gx, self.config.workgroup[1], self.config.workgroup[2])
        vk.vkEndCommandBuffer(command_buffer)

        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[command_buffer],
        )
        fence_info = vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
        fence = vk.vkCreateFence(device, fence_info, None)
        submit_start = _now_ms()
        vk.vkQueueSubmit(queue, 1, [submit_info], fence)
        vk.vkWaitForFences(device, 1, [fence], vk.VK_TRUE, 0xFFFFFFFFFFFFFFFF)
        vk.vkDestroyFence(device, fence, None)
        return _now_ms() - submit_start

    def _record_with_copies(
        self,
        *,
        device,
        command_buffer,
        pipeline,
        pipeline_layout,
        descriptor_set,
        queue,
        global_size: int,
        staging_in,
        device_in,
        staging_out,
        device_out,
        sign_nbytes: int,
        support_nbytes: int,
        dispatch_count: int,
        hash_only: bool,
        hash_buffer,
        hash_staging_buffer,
        hash_push_constant: int,
        hash_spv_path: Optional[Path],
        hash_handles: dict,
    ):
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(command_buffer, begin_info)

        # Copy staging inputs -> device-local inputs
        vk.vkCmdCopyBuffer(
            command_buffer,
            staging_in[0],
            device_in[0],
            1,
            [vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=sign_nbytes)],
        )
        vk.vkCmdCopyBuffer(
            command_buffer,
            staging_in[1],
            device_in[1],
            1,
            [vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=support_nbytes)],
        )

        # Barrier to make transfer writes visible to shader reads
        barriers_in = [
            vk.VkBufferMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                buffer=device_in[0],
                offset=0,
                size=sign_nbytes,
            ),
            vk.VkBufferMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                buffer=device_in[1],
                offset=0,
                size=support_nbytes,
            ),
        ]
        vk.vkCmdPipelineBarrier(
            command_buffer,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            None,
            len(barriers_in),
            barriers_in,
            0,
            None,
        )

        # Dispatch
        vk.vkCmdBindPipeline(command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
        vk.vkCmdBindDescriptorSets(
            command_buffer,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout,
            0,
            1,
            [descriptor_set],
            0,
            None,
        )
        gx = (global_size + self.config.workgroup[0] - 1) // self.config.workgroup[0]
        for _ in range(dispatch_count):
            vk.vkCmdDispatch(command_buffer, gx, self.config.workgroup[1], self.config.workgroup[2])

        # Barrier to make shader writes visible to transfer
        barriers_out = [
            vk.VkBufferMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                buffer=device_out[0],
                offset=0,
                size=sign_nbytes,
            ),
            vk.VkBufferMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                buffer=device_out[1],
                offset=0,
                size=support_nbytes,
            ),
        ]
        vk.vkCmdPipelineBarrier(
            command_buffer,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            None,
            len(barriers_out),
            barriers_out,
            0,
            None,
        )

        if hash_only:
            # Zero hash buffer then run hash reduction over outputs
            vk.vkCmdFillBuffer(command_buffer, hash_buffer, 0, vk.VK_WHOLE_SIZE, 0)
            hash_barrier_in = vk.VkBufferMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                buffer=hash_buffer,
                offset=0,
                size=vk.VK_WHOLE_SIZE,
            )
            vk.vkCmdPipelineBarrier(
                command_buffer,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                None,
                1,
                [hash_barrier_in],
                0,
                None,
            )
            hash_set_layout, hash_pipeline_layout = self._build_hash_layout(device)
            hash_shader = self._create_shader_module(device, hash_spv_path)
            hash_pipeline = self._create_hash_pipeline(device, hash_pipeline_layout, hash_shader)
            hash_descriptor_pool, hash_descriptor_set = self._allocate_hash_descriptors(device, hash_set_layout)
            hash_handles.update(
                {
                    "hash_set_layout": hash_set_layout,
                    "hash_pipeline_layout": hash_pipeline_layout,
                    "hash_shader": hash_shader,
                    "hash_pipeline": hash_pipeline,
                    "hash_descriptor_pool": hash_descriptor_pool,
                }
            )
            self._bind_hash_buffers(device, hash_descriptor_set, device_out[0], device_out[1], hash_buffer)
            vk.vkCmdBindPipeline(command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, hash_pipeline)
            vk.vkCmdBindDescriptorSets(
                command_buffer,
                vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                hash_pipeline_layout,
                0,
                1,
                [hash_descriptor_set],
                0,
                None,
            )
            vk.vkCmdPushConstants(
                command_buffer,
                hash_pipeline_layout,
                vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                4,
                np.int32(hash_push_constant).tobytes(),
            )
            gx_hash = (global_size + 256 - 1) // 256
            vk.vkCmdDispatch(command_buffer, gx_hash, 1, 1)
            # Barrier for hash writes -> transfer
            hash_barrier_out = vk.VkBufferMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                buffer=hash_buffer,
                offset=0,
                size=vk.VK_WHOLE_SIZE,
            )
            vk.vkCmdPipelineBarrier(
                command_buffer,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                None,
                1,
                [hash_barrier_out],
                0,
                None,
            )
            vk.vkCmdCopyBuffer(
                command_buffer,
                hash_buffer,
                hash_staging_buffer,
                1,
                [vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=4)],
            )
        else:
            # Copy outputs back to staging
            vk.vkCmdCopyBuffer(
                command_buffer,
                device_out[0],
                staging_out[0],
                1,
                [vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=sign_nbytes)],
            )
            vk.vkCmdCopyBuffer(
                command_buffer,
                device_out[1],
                staging_out[1],
                1,
                [vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=support_nbytes)],
            )

        vk.vkEndCommandBuffer(command_buffer)
        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[command_buffer],
        )
        fence_info = vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
        fence = vk.vkCreateFence(device, fence_info, None)
        submit_start = _now_ms()
        vk.vkQueueSubmit(queue, 1, [submit_info], fence)
        vk.vkWaitForFences(device, 1, [fence], vk.VK_TRUE, 0xFFFFFFFFFFFFFFFF)
        vk.vkDestroyFence(device, fence, None)
        return _now_ms() - submit_start


def build_vulkan_dispatcher(
    config: VulkanKernelConfig,
    dispatch_config: Optional[VulkanDispatchConfig] = None,
    *,
    shared_handles: Optional[VulkanHandles] = None,
) -> DispatchFn:
    """Factory that returns a callable dispatcher for use in VulkanBackendAdapter."""
    dispatcher = VulkanCarrierDispatcher(
        config=config,
        dispatch_config=dispatch_config,
        shared_handles=shared_handles,
    )
    return dispatcher.dispatch
