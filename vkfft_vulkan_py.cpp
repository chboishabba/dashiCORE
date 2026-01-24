// Minimal pybind11 bridge to DTolm/VkFFT (Vulkan backend).
// Exposes VkFFTPlan that records a single FFT/iFFT into a fresh command buffer,
// submits it to the provided queue, and waits via a fence.
//
// Build with `setup_vkfft_vulkan.py`. Expects VkFFT headers available
// (clone https://github.com/DTolm/VkFFT and point include path there).

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef VKFFT_BACKEND
#define VKFFT_BACKEND 0  // Vulkan
#endif
#include "vkFFT.h"

namespace py = pybind11;

static uint64_t as_u64(py::handle h) {
    if (py::isinstance<py::int_>(h)) return h.cast<uint64_t>();
    if (py::hasattr(h, "handle")) return py::getattr(h, "handle").cast<uint64_t>();
    if (py::hasattr(h, "value")) return py::getattr(h, "value").cast<uint64_t>();
    throw std::runtime_error("Expected an int-like Vulkan handle (or object with .handle/.value)");
}

static void vkfft_check(VkFFTResult r, const char* what) {
    if (r != VKFFT_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed: VkFFTResult=" + std::to_string((int)r));
    }
}

struct PyVkFFTPlan {
    VkPhysicalDevice physical_device{};
    VkDevice device{};
    VkQueue queue{};
    VkCommandPool command_pool{};
    VkFence fence{};

    VkBuffer buffer{};
    VkBuffer scratch_buffer{};
    uint64_t buffer_size{};
    uint64_t scratch_size{};
    uint32_t queue_family_index{};

    int inverse_flag{-1};  // -1 = forward, 1 = inverse (VkFFT convention)
    VkFFTApplication app{};
    VkFFTConfiguration cfg{};
    VkFFTLaunchParams launch{};
    bool inited{false};

    PyVkFFTPlan(uint64_t physical_device_h,
                uint64_t device_h,
                uint64_t queue_h,
                uint64_t command_pool_h,
                uint32_t queue_family_index_in,
                uint64_t buffer_size_bytes,
                uint64_t scratch_size_bytes,
                const std::vector<uint64_t>& dims,
                uint64_t buffer_h,
                uint64_t scratch_h,
                bool inverse) {
        physical_device = reinterpret_cast<VkPhysicalDevice>(physical_device_h);
        device = reinterpret_cast<VkDevice>(device_h);
        queue = reinterpret_cast<VkQueue>(queue_h);
        command_pool = reinterpret_cast<VkCommandPool>(command_pool_h);
        queue_family_index = queue_family_index_in;

        buffer = reinterpret_cast<VkBuffer>(buffer_h);
        scratch_buffer = reinterpret_cast<VkBuffer>(scratch_h);
        buffer_size = buffer_size_bytes;
        scratch_size = scratch_size_bytes;
        inverse_flag = inverse ? 1 : -1;

        std::memset(&app, 0, sizeof(app));
        std::memset(&cfg, 0, sizeof(cfg));
        std::memset(&launch, 0, sizeof(launch));

        cfg.physicalDevice = &physical_device;
        cfg.device = &device;
        cfg.queue = &queue;
        cfg.commandPool = &command_pool;
        cfg.fence = &fence;
        cfg.FFTdim = static_cast<uint64_t>(dims.size());
        for (size_t i = 0; i < dims.size() && i < 3; i++) cfg.size[i] = dims[i];

        cfg.buffer = &buffer;
        cfg.bufferSize = &buffer_size;
        if (scratch_h != 0) cfg.tempBuffer = &scratch_buffer;
        if (scratch_h != 0) cfg.tempBufferSize = &scratch_size;
        // Build a bidirectional plan to avoid ONLY_FORWARD/ONLY_INVERSE errors.
        cfg.makeForwardPlanOnly = 0;
        cfg.makeInversePlanOnly = 0;

        // Create fence for VkFFT internal use.
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(device, &fci, nullptr, &fence) != VK_SUCCESS) {
            throw std::runtime_error("vkCreateFence failed");
        }

        vkfft_check(initializeVkFFT(&app, cfg), "initializeVkFFT");
        inited = true;
    }

    ~PyVkFFTPlan() {
        if (inited) {
            deleteVkFFT(&app);
            inited = false;
        }
        if (fence) {
            vkDestroyFence(device, fence, nullptr);
            fence = VK_NULL_HANDLE;
        }
    }

    void exec() {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = command_pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;

        VkCommandBuffer cmd{};
        if (vkAllocateCommandBuffers(device, &ai, &cmd) != VK_SUCCESS) {
            throw std::runtime_error("vkAllocateCommandBuffers failed");
        }

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
            throw std::runtime_error("vkBeginCommandBuffer failed");
        }

        launch.commandBuffer = &cmd;
        vkfft_check(VkFFTAppend(&app, inverse_flag, &launch), "VkFFTAppend");

        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
            throw std::runtime_error("vkEndCommandBuffer failed");
        }

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;

        // Reset fence in case exec() is called multiple times.
        vkResetFences(device, 1, &fence);
        if (vkQueueSubmit(queue, 1, &si, fence) != VK_SUCCESS) {
            throw std::runtime_error("vkQueueSubmit failed");
        }

        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

        vkFreeCommandBuffers(device, command_pool, 1, &cmd);
    }
};

PYBIND11_MODULE(vkfft_vulkan_py, m) {
    py::class_<PyVkFFTPlan>(m, "VkFFTPlan")
        .def(py::init([](py::object physical_device,
                         py::object device,
                         py::object queue,
                         py::object command_pool,
                         uint32_t queue_family_index,
                         uint64_t buffer_size_bytes,
                         uint64_t scratch_size_bytes,
                         std::vector<uint64_t> dims,
                         py::object buffer,
                         py::object scratch_buffer,
                         bool inverse) {
            return new PyVkFFTPlan(
                as_u64(physical_device),
                as_u64(device),
                as_u64(queue),
                as_u64(command_pool),
                queue_family_index,
                buffer_size_bytes,
                scratch_size_bytes,
                dims,
                as_u64(buffer),
                scratch_buffer.is_none() ? 0ULL : as_u64(scratch_buffer),
                inverse);
        }),
        py::arg("physical_device"),
        py::arg("device"),
        py::arg("queue"),
        py::arg("command_pool"),
        py::arg("queue_family_index"),
        py::arg("buffer_size_bytes"),
        py::arg("scratch_size_bytes") = 0,
        py::arg("dims"),
        py::arg("buffer"),
        py::arg("scratch_buffer") = py::none(),
        py::arg("inverse") = false)
        .def("exec", &PyVkFFTPlan::exec);
}
