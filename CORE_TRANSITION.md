# CORE → GPU Transition Notes

This note captures how we already exercise GPUs across the repo and how to apply those patterns to `dashiCORE`. It also plants adjacent GPU helper files (prefixed `gpu_*.py`) so you can start wiring a Vulkan-backed kernel without re-scavenging code.

## Known GPU implementations to crib
- Vulkan compute prototypes: `vulkan_compute/compute_buffer.py`, `vulkan_compute/compute_image.py`, `vulkan_compute/compute_image_preview.py` (storage buffers/images, SPIR-V via `glslc`).
- Vulkan video path: `vulkan/video_bench_vk.py` + `vulkan/README.md` (swapchain + optional compute pass; VAAPI/dmabuf notes).
- Trading qfeat tape on GPU: `trading/vk_qfeat.py` (GLSL → SPIR-V compile, SSBO wiring, timing output).
- GPU/JAX parity map: `docs/vulkan_jax_parity.md` (first kernel suggestion: blockwise residual/diff).
- JAX prototypes: `JAX/*.py` are GPU-friendly references, but ROCm on gfx803/RX 580 has been flaky; expect CPU fallback unless ROCm is restored.

## Copied helpers (adjacent `gpu_*.py`)
- `gpu_common_methods.py` now holds:
  - `compile_shader(...)` copied from `trading/vk_qfeat._compile_shader` (GLSL → SPIR-V via `glslc`, supports `-D` defines).
  - `find_memory_type(...)` copied from `vulkan_compute/compute_buffer._find_memory_type` (host-visible/host-coherent memory selection).
- When adding GPU kernels near `dashi_core/`, mirror this naming (`gpu_<purpose>.py`) to keep the lineage obvious and avoid touching the pure core modules.
- `gpu_vulkan_adapter.py` exposes `VulkanBackendAdapter` + `VulkanCarrierKernel`:
  - Configurable compile on init or dispatch; skips `glslc` when SPIR-V is newer than GLSL.
  - Inject a dispatcher callable to run the GPU path; if missing and `allow_fallback=False`, the adapter raises instead of silently no-oping.
  - All dispatches validate `Carrier` invariants post-run; support creation or shape drift is rejected.
- `gpu_vulkan_backend.py` adds a registry-friendly `VulkanBackend` plus `register_vulkan_backend(...)` and `make_vulkan_kernel(...)` helpers to slot the adapter into `dashi_core.backend.use_backend(...)` parity tests without touching core modules.
- `gpu_vulkan_dispatcher.py` implements a minimal Vulkan compute dispatcher (storage-buffer pipeline) using a GLSL/ SPIR-V shader (see `gpu_shaders/carrier_passthrough.comp`); builds a dispatcher with `build_vulkan_dispatcher(...)` and wires it into `register_vulkan_backend(...)` by default.
- Vulkan binding quirks observed on RADV/RX 580 (python-vulkan + cffi):
  - Use `apiVersion=vk.VK_MAKE_VERSION(1, 2, 0)` (not `VK_API_VERSION_1_2`) for portability across bindings.
  - `vkMapMemory` may return int, bytes-like, or cffi buffers; `_write_buffer` / `_read_buffer` normalize via memoryview first, then fall back to pointer arithmetic. Avoid casting mapped buffers directly to int.
  - `VkShaderModuleCreateInfo` accepts raw SPIR-V bytes for `pCode`; passing numpy lists can break some bindings.
  - Working env/launch used for validation: `export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json`, `glslc gpu_shaders/carrier_passthrough.comp -o gpu_shaders/carrier_passthrough.spv`, then `python scripts/run_vulkan_passthrough.py` or `python scripts/run_backend_parity_example.py`.

## SPIR-V build/run patterns (Vulkan)
- Compile GLSL to SPIR-V (compute):
  - `glslc vulkan_compute/shaders/add.comp -o vulkan_compute/shaders/add.spv`
  - `glslc vulkan_compute/shaders/write_image.comp -o vulkan_compute/shaders/write_image.spv`
  - For video diff: `glslc vulkan/shaders/diff.comp -o vulkan/shaders/diff.spv`
- Typical env for RADV/RX 580: `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json`.
- Minimal run loops:
  - Storage buffer sample: `python vulkan_compute/compute_buffer.py`
  - Storage image sample (optional dump): `python vulkan_compute/compute_image.py --dump out.ppm`
  - Live preview: `python vulkan_compute/compute_image_preview.py --width 512 --height 512 --frames 240`
  - Trading qfeat GPU path: `python -m trading.vk_qfeat ... --backend vk --shader trading/shaders/qfeat.comp --spv trading/shaders/qfeat.spv`

## Transition steps for `dashiCORE`
1. **Keep core pure.** New GPU modules should live beside (not inside) `dashi_core/` and import its `Carrier`/`Kernel` datatypes without mutating them. Name them `gpu_<role>.py`.
2. **Reuse helpers.** Start from `gpu_common_methods.py` for SPIR-V compilation and memory selection; keep Vulkan host/device plumbing minimal and explicit.
3. **Respect invariants.** Kernels must be shape-preserving and must not create new support; map GPU outputs back into `Carrier` via `Carrier.from_signed` or explicit `support/sign` construction.
4. **Backend selection.** Thread `VK_ICD_FILENAMES` and shader paths via flags/env; keep a CPU fallback path to stay usable when Vulkan is unavailable.
5. **Testing.** Mirror `dashiCORE/TESTING.md` invariants in GPU tests (support preservation, defect ≥ 0). Borrow fixtures from `vulkan_compute/tests` once they exist; for now, smoke-test dispatch with small buffers/images.
6. **JAX note.** gfx803/RX 580 has spotty ROCm support; do not assume JAX-on-GPU. Treat JAX modules as math references only unless ROCm is confirmed.

## Immediate next hooks
- Add a thin Vulkan kernel adapter (e.g., `gpu_kernel.py`) that:
  - Compiles a GLSL compute shader (see `compile_shader` helper).
  - Binds input/output SSBOs for `Carrier` data (support + sign).
  - Dispatches with fixed workgroup sizes and returns a new `Carrier`.
- Document any new shader contracts next to their SPIR-V outputs to keep the GPU path reproducible.
- Test device note: a Vulkan-capable GPU is available for validation here; set `VK_ICD_FILENAMES` accordingly and run `python scripts/run_vulkan_core_mask.py` or `python scripts/run_vulkan_passthrough.py` to exercise the path.
- FFT placeholder: `gpu_vkfft_stub.py` provides `fft2/ifft2/has_vkfft` that fall back to NumPy with a warning when vkFFT bindings are absent; swap to real vkFFT when bindings are available on the target GPU.
- Majority fusion shader: `gpu_shaders/core_mask_majority.comp` (+ `.spv`) implements K-channel majority vote on ternary carriers (channel-major layout, push constants `n` and `k`, tie → support=0). Compile via `glslc gpu_shaders/core_mask_majority.comp -o gpu_shaders/core_mask_majority.spv`.
- ICD probing: when `VK_ICD_FILENAMES` is unset, a robust loop is `for icd in /usr/share/vulkan/icd.d/*.json /etc/vulkan/icd.d/*.json; do [ -f \"$icd\" ] && VK_ICD_FILENAMES=\"$icd\" python scripts/run_vulkan_core_mask.py && break; done`.
