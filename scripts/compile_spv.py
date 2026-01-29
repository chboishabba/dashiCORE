#!/usr/bin/env python3
"""
Compile all GLSL compute shaders from dashiCORE/spv/comp into dashiCORE/spv.

Defaults to compile-on-change (skips if .spv is newer than .comp).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_common_methods import compile_shader  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compile dashiCORE/spv/comp/*.comp -> dashiCORE/spv/*.spv")
    ap.add_argument("--force", action="store_true", help="Recompile even if SPV is newer")
    ap.add_argument(
        "--include-legacy",
        action="store_true",
        help="Also compile dashiCORE/gpu_shaders/*.comp into dashiCORE/spv/",
    )
    return ap.parse_args()


def _compile_dir(comp_dir: Path, out_dir: Path, force: bool) -> int:
    compiled = 0
    for shader in sorted(comp_dir.glob("*.comp")):
        spv = out_dir / f"{shader.stem}.spv"
        if force and spv.exists():
            spv.unlink()
        compile_shader(shader, spv)
        compiled += 1
    return compiled


def main() -> None:
    args = parse_args()
    comp_dir = ROOT / "spv" / "comp"
    out_dir = ROOT / "spv"
    if not comp_dir.is_dir():
        raise SystemExit(f"Missing comp dir: {comp_dir}")
    total = _compile_dir(comp_dir, out_dir, args.force)
    if args.include_legacy:
        legacy_dir = ROOT / "gpu_shaders"
        if legacy_dir.is_dir():
            total += _compile_dir(legacy_dir, out_dir, args.force)
    print(f"[spv] compiled {total} shaders into {out_dir}")


if __name__ == "__main__":
    main()
