from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Dict, List


def _parse_size(text: str) -> int:
    """Parse sizes like '32K', '256K', '3M'. Return bytes or 0 on failure."""
    m = re.match(r"^\s*(\d+)\s*([KMG])?\s*$", text.strip(), re.I)
    if not m:
        return 0
    n = int(m.group(1))
    unit = (m.group(2) or "").upper()
    mult = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}.get(unit, 1)
    return n * mult


def cpu_cache_info() -> Dict[str, int]:
    """Best-effort CPU cache info (bytes, line size, page size)."""
    base = Path("/sys/devices/system/cpu/cpu0/cache")
    info: Dict[str, int] = {}
    if base.exists():
        for idx in base.glob("index*"):
            try:
                level = (idx / "level").read_text().strip()
                size = (idx / "size").read_text().strip()
                typ = (idx / "type").read_text().strip()
                info[f"L{level}_{typ}"] = _parse_size(size)
            except OSError:
                continue
        try:
            info["line_size"] = int((base / "index0" / "coherency_line_size").read_text().strip())
        except OSError:
            pass
    try:
        info["page_size"] = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        pass
    return info


def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b) if a and b else 0


def recommend_pq_block_elems(
    *,
    workgroup_size: int = 256,
    cacheline_bytes: int = 64,
    l1_bytes: int = 32 * 1024,
    l2_bytes: int = 256 * 1024,
    max_block_elems: int = 1_048_576,
) -> List[int]:
    """
    Heuristic PQ block sizes (in elements) aligned to cache lines + workgroups.
    PQ packs 4 elements per byte (2 bits/element).
    """
    elems_per_cacheline_pq = cacheline_bytes * 4
    align = lcm(lcm(4, workgroup_size), elems_per_cacheline_pq)
    targets_pq_bytes = sorted(
        set(
            [
                max(256, l1_bytes // 8),
                max(256, l1_bytes // 4),
                max(512, l2_bytes // 16),
                max(512, l2_bytes // 8),
            ]
        )
    )

    candidates = set()
    for t_bytes in targets_pq_bytes:
        base_elems = t_bytes * 4
        elems = ((base_elems + align - 1) // align) * align
        for mul in [1, 2, 4, 8]:
            e = elems * mul
            if e <= max_block_elems:
                candidates.add(e)

    for e in [workgroup_size * k for k in [1, 2, 4, 8, 16, 32]]:
        if e <= max_block_elems:
            candidates.add(((e + align - 1) // align) * align)

    return sorted(candidates)
