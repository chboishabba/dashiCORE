from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

from gpu_common_methods import compile_shader, resolve_shader_candidates, resolve_spv
from gpu_vulkan_dispatcher import (
    HOST_VISIBLE_COHERENT,
    VulkanDispatchConfig,
    VulkanHandles,
    _create_buffer,
    _create_device_and_queue,
    _pick_compute_queue_family,
    _ptr_to_int,
    _read_buffer,
    _require_vk,
    _write_buffer,
    create_vulkan_handles,
    find_memory_type,
)

try:
    import vulkan as vk
except ImportError:  # pragma: no cover
    vk = None  # type: ignore


class VulkanGemvExecutor:
    """
    Minimal Vulkan GEMV (y = A*x) executor backed by a SPIR-V compute shader.

    Notes:
    - Float32 only (inputs are converted to float32); outputs returned as float32.
    - Host-visible buffers for simplicity; intended for benchmarking correctness/perf, not ultimate speed.
    - Reuses device/pipeline/buffers across calls for the same N.
    """

    def __init__(
        self,
        N: int,
        *,
        handles: Optional[VulkanHandles] = None,
        workgroup_size: int = 256,
        dispatch_config: Optional[VulkanDispatchConfig] = None,
        shader_path: Optional[Path] = None,
        timing_enabled: bool = True,
    ):
        _require_vk()
        self.N = int(N)
        self.workgroup_size = int(workgroup_size)
        self.dispatch_config = dispatch_config or VulkanDispatchConfig()
        if shader_path is None:
            shader_path = resolve_shader_candidates(("gemv_tiled", "gemv"))
        self.shader_path = shader_path
        self.spv_path = resolve_spv(shader_path.stem)
        compile_shader(self.shader_path, self.spv_path)

        self.external_handles = handles
        self.handles = handles or create_vulkan_handles(self.dispatch_config)
        self._owns_handles = handles is None
        self.timing_enabled = bool(timing_enabled)

        self.device = self.handles.device
        self.queue = self.handles.queue
        self.mem_props = self.handles.mem_props
        self.queue_family_index = self.handles.queue_family_index

        self._build_pipeline()
        self._allocate_buffers()
        self._timing_last = {
            "gpu_wait_ms": 0.0,
            "fence_wait_ms": 0.0,
        }

    def _timing_reset(self) -> None:
        if not self.timing_enabled:
            return
        self._timing_last = {
            "gpu_wait_ms": 0.0,
            "fence_wait_ms": 0.0,
        }

    def get_last_timings(self) -> dict:
        return dict(self._timing_last) if self.timing_enabled else {}

    def _build_pipeline(self) -> None:
        device = self.device

        # Descriptor set layout: binding 0 = A, 1 = x, 2 = y
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
        self.descriptor_set_layout = vk.vkCreateDescriptorSetLayout(device, layout_info, None)

        push_constant_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=4,  # uint N
        )
        pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[self.descriptor_set_layout],
            pushConstantRangeCount=1,
            pPushConstantRanges=[push_constant_range],
        )
        self.pipeline_layout = vk.vkCreatePipelineLayout(device, pipeline_layout_info, None)

        shader_code = Path(self.spv_path).read_bytes()
        module_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(shader_code),
            pCode=shader_code,
        )
        self.shader_module = vk.vkCreateShaderModule(device, module_info, None)

        stage_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=self.shader_module,
            pName="main",
        )
        pipeline_info = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage_info,
            layout=self.pipeline_layout,
        )
        self.pipeline = vk.vkCreateComputePipelines(device, vk.VK_NULL_HANDLE, 1, [pipeline_info], None)[0]

        pool_sizes = [
            vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=3),
        ]
        pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=1,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
        )
        self.descriptor_pool = vk.vkCreateDescriptorPool(device, pool_info, None)
        alloc_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[self.descriptor_set_layout],
        )
        self.descriptor_set = vk.vkAllocateDescriptorSets(device, alloc_info)[0]

    def _allocate_buffers(self) -> None:
        device = self.device
        mem_props = self.mem_props
        N = self.N
        nbytes_mat = N * N * 4  # float32
        nbytes_vec = N * 4
        usage = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        flags = HOST_VISIBLE_COHERENT

        self.buf_A, self.mem_A = _create_buffer(device, mem_props, nbytes_mat, usage, flags)
        self.buf_x, self.mem_x = _create_buffer(device, mem_props, nbytes_vec, usage, flags)
        self.buf_y, self.mem_y = _create_buffer(device, mem_props, nbytes_vec, usage, flags)

        # Bind buffers to descriptor set
        write_sets = [
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self.descriptor_set,
                dstBinding=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[
                    vk.VkDescriptorBufferInfo(buffer=self.buf_A, offset=0, range=nbytes_mat)
                ],
            ),
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self.descriptor_set,
                dstBinding=1,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[
                    vk.VkDescriptorBufferInfo(buffer=self.buf_x, offset=0, range=nbytes_vec)
                ],
            ),
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self.descriptor_set,
                dstBinding=2,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[
                    vk.VkDescriptorBufferInfo(buffer=self.buf_y, offset=0, range=nbytes_vec)
                ],
            ),
        ]
        vk.vkUpdateDescriptorSets(self.device, len(write_sets), write_sets, 0, None)

        # Command pool/buffer
        pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=self.queue_family_index,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        )
        self.command_pool = vk.vkCreateCommandPool(self.device, pool_info, None)
        alloc_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        self.command_buffer = vk.vkAllocateCommandBuffers(self.device, alloc_info)[0]

    def gemv(self, A: np.ndarray, x: np.ndarray) -> np.ndarray:
        self._timing_reset()
        # Inputs are converted to float32 row-major.
        A32 = np.asarray(A, dtype=np.float32, order="C")
        x32 = np.asarray(x, dtype=np.float32, order="C")
        if A32.shape[0] != self.N or A32.shape[1] != self.N:
            raise ValueError(f"A shape {A32.shape} does not match executor N={self.N}")
        if x32.shape[0] != self.N:
            raise ValueError(f"x shape {x32.shape} does not match executor N={self.N}")

        _write_buffer(self.device, self.mem_A, A32)
        _write_buffer(self.device, self.mem_x, x32)

        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(self.command_buffer, begin_info)
        vk.vkCmdBindPipeline(self.command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline)
        vk.vkCmdBindDescriptorSets(
            self.command_buffer,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline_layout,
            0,
            1,
            [self.descriptor_set],
            0,
            None,
        )
        if hasattr(vk, "ffi"):
            push_data = vk.ffi.new("uint32_t[]", [self.N])
        else:
            push_data = np.array([self.N], dtype=np.uint32)
        vk.vkCmdPushConstants(
            self.command_buffer,
            self.pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            4,
            push_data,
        )
        groups = math.ceil(self.N / self.workgroup_size)
        vk.vkCmdDispatch(self.command_buffer, groups, 1, 1)
        vk.vkEndCommandBuffer(self.command_buffer)

        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[self.command_buffer],
        )
        fence = vk.vkCreateFence(self.device, vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO), None)
        vk.vkQueueSubmit(self.queue, 1, [submit_info], fence)
        t0 = time.perf_counter()
        vk.vkWaitForFences(self.device, 1, [fence], vk.VK_TRUE, 1_000_000_000)
        wait_ms = 1000 * (time.perf_counter() - t0)
        if self.timing_enabled:
            self._timing_last["fence_wait_ms"] += wait_ms
            self._timing_last["gpu_wait_ms"] += wait_ms
        vk.vkDestroyFence(self.device, fence, None)

        y = _read_buffer(self.device, self.mem_y, (self.N,), np.float32)
        vk.vkResetCommandBuffer(self.command_buffer, 0)
        return y

    def close(self) -> None:
        device = self.device
        if not device:
            return
        if hasattr(self, "buf_A"):
            vk.vkDestroyBuffer(device, self.buf_A, None)
        if hasattr(self, "buf_x"):
            vk.vkDestroyBuffer(device, self.buf_x, None)
        if hasattr(self, "buf_y"):
            vk.vkDestroyBuffer(device, self.buf_y, None)
        if hasattr(self, "mem_A"):
            vk.vkFreeMemory(device, self.mem_A, None)
        if hasattr(self, "mem_x"):
            vk.vkFreeMemory(device, self.mem_x, None)
        if hasattr(self, "mem_y"):
            vk.vkFreeMemory(device, self.mem_y, None)
        if hasattr(self, "command_pool"):
            vk.vkDestroyCommandPool(device, self.command_pool, None)
        if hasattr(self, "pipeline"):
            vk.vkDestroyPipeline(device, self.pipeline, None)
        if hasattr(self, "pipeline_layout"):
            vk.vkDestroyPipelineLayout(device, self.pipeline_layout, None)
        if hasattr(self, "shader_module"):
            vk.vkDestroyShaderModule(device, self.shader_module, None)
        if hasattr(self, "descriptor_pool"):
            vk.vkDestroyDescriptorPool(device, self.descriptor_pool, None)
        if hasattr(self, "descriptor_set_layout"):
            vk.vkDestroyDescriptorSetLayout(device, self.descriptor_set_layout, None)
        if self._owns_handles and self.handles is not None:
            self.handles.close()


def has_vulkan() -> bool:
    try:
        _require_vk()
        return True
    except Exception:
        return False
