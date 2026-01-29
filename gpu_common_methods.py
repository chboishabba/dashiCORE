from __future__ import annotations

import os
import pathlib
from typing import Any, Iterable, Optional, Sequence

CORE_ROOT = pathlib.Path(__file__).resolve().parent
SPV_COMP_DIR = CORE_ROOT / "spv" / "comp"
SPV_DIR = CORE_ROOT / "spv"
LEGACY_SHADER_DIR = CORE_ROOT / "gpu_shaders"


def resolve_shader(name: str) -> pathlib.Path:
    """
    Resolve a GLSL compute shader by name.

    Prefers dashiCORE/spv/comp/{name}.comp, falls back to dashiCORE/gpu_shaders/{name}.comp.
    """
    preferred = SPV_COMP_DIR / f"{name}.comp"
    if preferred.exists():
        return preferred
    legacy = LEGACY_SHADER_DIR / f"{name}.comp"
    if legacy.exists():
        return legacy
    return preferred


def resolve_shader_candidates(names: Sequence[str]) -> pathlib.Path:
    """
    Resolve the first available shader from a list of candidate names.
    """
    last = None
    for name in names:
        path = resolve_shader(name)
        last = path
        if path.exists():
            return path
    if last is None:
        raise ValueError("resolve_shader_candidates called with empty names")
    return last


def resolve_spv(name: str) -> pathlib.Path:
    """
    Resolve SPIR-V output path for a shader name into dashiCORE/spv/{name}.spv.
    """
    return SPV_DIR / f"{name}.spv"


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
    spv_path.parent.mkdir(parents=True, exist_ok=True)
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
