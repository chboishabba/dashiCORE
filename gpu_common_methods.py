from __future__ import annotations

import os
import pathlib
from typing import Any, Iterable, Optional


def compile_shader(
    shader_path: pathlib.Path,
    spv_path: pathlib.Path,
    *,
    defines: Optional[Iterable[str]] = None,
) -> None:
    """
    GLSL → SPIR-V compile helper (copied from trading/vk_qfeat._compile_shader).
    Skips compilation when the SPIR-V file is newer than the GLSL source.
    """
    if not shader_path.exists():
        raise FileNotFoundError(shader_path)
    if spv_path.exists() and spv_path.stat().st_mtime >= shader_path.stat().st_mtime:
        return
    define_args = [f"-D{define}" for define in defines or []]
    cmd = ["glslc", *define_args, str(shader_path), "-o", str(spv_path)]
    result = os.spawnvp(os.P_WAIT, "glslc", cmd)
    if result != 0:
        raise RuntimeError(f"glslc failed with exit code {result}")


def find_memory_type(
    mem_props: Any,
    type_bits: int,
    required_flags: int,
) -> int:
    """
    Host-visible/coherent memory selector (copied from vulkan_compute/compute_buffer._find_memory_type).
    Expects a VkPhysicalDeviceMemoryProperties-like object.
    """
    for idx in range(mem_props.memoryTypeCount):
        if type_bits & (1 << idx):
            flags = mem_props.memoryTypes[idx].propertyFlags
            if (flags & required_flags) == required_flags:
                return idx
    raise RuntimeError("No compatible memory type found.")
