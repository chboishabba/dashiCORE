from __future__ import annotations

import os
from pathlib import Path
from typing import List

from setuptools import Extension, setup

try:
    import pybind11
except ImportError as exc:  # pragma: no cover - setup-only import
    raise SystemExit("pybind11 is required to build vkfft_vulkan_py") from exc


def _vkfft_include_dirs() -> List[str]:
    """Resolve VkFFT include path from env or common locations."""
    env = os.environ.get("VKFFT_INCLUDE_DIR")
    if env:
        return [env, os.path.join(env, "vkFFT")]
    candidates = [
        Path("third_party/VkFFT"),
        Path("VkFFT"),
    ]
    includes: List[str] = []
    for base in candidates:
        if base.joinpath("vkFFT.h").exists():
            includes.append(str(base))
        sub = base / "vkFFT"
        if sub.joinpath("vkFFT.h").exists():
            includes.append(str(sub))
    return includes


def _glslang_include_dirs() -> List[str]:
    """Common glslang include locations; extend if your distro differs."""
    env = os.environ.get("GLSLANG_INCLUDE_DIR")
    if env:
        return [env, os.path.join(env, "Include")]
    candidates = [
        Path("/usr/include/glslang"),
        Path("/usr/include/glslang/Include"),
        Path("/usr/local/include/glslang"),
        Path("/usr/local/include/glslang/Include"),
    ]
    return [str(p) for p in candidates if p.exists()]


vkfft_includes = _vkfft_include_dirs()
glslang_includes = _glslang_include_dirs()
if not vkfft_includes:
    print("warning: vkFFT.h not found. Set VKFFT_INCLUDE_DIR or clone https://github.com/DTolm/VkFFT into third_party/VkFFT")
if not glslang_includes:
    print("warning: glslang headers not found (need glslang_c_interface.h); install glslang-dev or set GLSLANG_INCLUDE_DIR")

ext = Extension(
    "vkfft_vulkan_py",
    sources=["vkfft_vulkan_py.cpp"],
    include_dirs=[pybind11.get_include(), *vkfft_includes, *glslang_includes],
    language="c++",
    extra_compile_args=["-std=c++17", "-DVKFFT_BACKEND=0"],
    libraries=[
        "vulkan",
        "glslang",
        "SPIRV",
        "glslang-default-resource-limits",
        "SPIRV-Tools-opt",
        "SPIRV-Tools",
    ],
)

setup(name="vkfft_vulkan_py", version="0.0.0", ext_modules=[ext])
