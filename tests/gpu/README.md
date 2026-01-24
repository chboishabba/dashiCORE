# GPU Tests

These tests assert that GPU adapters/backends stay outside `dashi_core/`, enforce the same Carrier invariants, and can be registered into the backend registry without changing semantics.

- `test_vulkan_adapter.py` exercises compile hooks, dispatcher routing, and forbids support creation or shape drift.
- `test_vulkan_backend_registry.py` registers a Vulkan backend via helper utilities and checks parity with CPU (via fallback) plus dispatcher invocation.
- `test_vulkan_passthrough_parity.py` is a canary parity test (skips unless `VK_ICD_FILENAMES` + `glslc` + python-vulkan are available); it runs the passthrough shader on a real Vulkan backend and asserts CPU/Vulkan equality.
- `test_vulkan_sign_flip.py` runs a non-trivial sign-flip shader, asserts parity/idempotence, and preserves support.
- `test_vulkan_two_pass.py` composes two GPU dispatches (flip then clamp) to validate ordering and determinism.
- `test_vulkan_repeatability.py` stresses determinism by running the sign-flip shader many times and asserting identical outputs.

The sample shader `gpu_shaders/carrier_passthrough.comp` matches the storage-buffer layout expected by `gpu_vulkan_dispatcher.py`. When running on a GPU, build it with `glslc gpu_shaders/carrier_passthrough.comp -o gpu_shaders/carrier_passthrough.spv` and pass those paths into `VulkanKernelConfig`.

> GPU paths must be observationally equivalent to CPU. Any support creation, nondeterministic output, or shape change is a backend bug.

## Vulkan binding quirks (python-vulkan + RADV RX 580)

- API version: set `apiVersion=vk.VK_MAKE_VERSION(1, 2, 0)`; some bindings omit `VK_API_VERSION_1_2`.
- Mapped memory: `vkMapMemory` can return ints, bytes-like, or cffi buffers. The dispatcher normalizes via `memoryview` first and only falls back to pointer arithmetic; avoid casting mapped buffers to int directly.
- Shader module: pass raw SPIR-V bytes (`pCode=code_bytes`) into `VkShaderModuleCreateInfo`; lists/arrays can break some bindings.
- Working env/run steps validated here:
  - `export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json`
  - `glslc gpu_shaders/carrier_passthrough.comp -o gpu_shaders/carrier_passthrough.spv`
  - `python scripts/run_vulkan_passthrough.py`
