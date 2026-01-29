#!/usr/bin/env bash

# VkFFT + Vulkan build/runtime environment
source /Whisper-WebUI/venv/bin/activate
export VKFFT_INCLUDE_DIR="/opt/dashiCFD/dashiCORE/third_party/VkFFT"
export GLSLANG_INCLUDE_DIR="/usr/include/glslang/Include"

# Linker/runtime paths for injected libs
export LIBRARY_PATH="/usr/local/lib/dashi:/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="/usr/local/lib/dashi:/usr/lib/x86_64-linux-gnu"
export LDFLAGS="-L/usr/local/lib/dashi -L/usr/lib/x86_64-linux-gnu"
export PYTHONPATH="/opt/dashiCFD/dashiCORE:${PYTHONPATH}"
