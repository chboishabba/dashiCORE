"""
Standalone pyvkfft OpenCL roundtrip with explicit platform/device logging.
Run with system python (no venv) to avoid mixed ABI stacks:
    python delme.py
Set PYOPENCL_CTX=\"<plat>:<dev>\" to pick a device explicitly.
"""

import faulthandler
import os
import sys
import numpy as np
import pyopencl as cl
from pyvkfft.opencl import VkFFTApp


def pick_context() -> cl.Context:
    """Prefer PYOPENCL_CTX; otherwise pick the first enumerated device."""
    try:
        return cl.create_some_context(interactive=False)
    except Exception:
        platforms = cl.get_platforms()
        devices = [d for p in platforms for d in p.get_devices()]
        if not devices:
            raise RuntimeError("No OpenCL devices found (check /dev/dri permissions)")
        return cl.Context([devices[0]])


def main() -> None:
    faulthandler.enable()

    platforms = cl.get_platforms()
    print(f"platforms ({len(platforms)}):")
    for pi, p in enumerate(platforms):
        print(f"  [{pi}] {p.name} ({p.vendor})")
        for di, d in enumerate(p.get_devices()):
            print(f"     device[{di}]: {d.name}  type={cl.device_type.to_string(d.type)}")

    print(f"PYOPENCL_CTX={os.environ.get('PYOPENCL_CTX', '<unset>')}")
    ctx = pick_context()
    device = ctx.devices[0]
    print(f"using device: {device.name} ({cl.device_type.to_string(device.type)})")

    queue = cl.CommandQueue(ctx)

    x = np.random.rand(16, 16).astype(np.complex64)
    buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)

    app = VkFFTApp(x.shape, x.dtype, queue=queue)
    app.fft(buf)
    app.ifft(buf)

    out = np.empty_like(x)
    cl.enqueue_copy(queue, out, buf).wait()
    max_err = float(np.max(np.abs(out - x)))
    print("roundtrip ok; max error:", max_err)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
