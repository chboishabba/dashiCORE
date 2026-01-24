#!/usr/bin/env python3
"""Coverage guard for benchmarks.

Ensures public exports in dashi_core are listed in benchmarks/coverage.yaml and that
referenced symbols actually exist.
"""

from __future__ import annotations

import sys
import yaml
from importlib import import_module
from pathlib import Path
from typing import Dict, Set


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_plan(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _public_exports() -> Set[str]:
    import dashi_core as dc

    return set(getattr(dc, "__all__", []))


def _module_symbols(mod_name: str) -> Set[str]:
    mod = import_module(mod_name)
    return set(dir(mod))


def main() -> int:
    plan_path = ROOT / "benchmarks" / "coverage.yaml"
    if not plan_path.exists():
        print("coverage.yaml missing", file=sys.stderr)
        return 1
    plan = load_plan(plan_path)
    exports = set(plan.get("exports", []))
    public = _public_exports()

    missing_from_plan = public - exports
    extra_in_plan = exports - public

    missing_symbols = []
    for mod, entries in plan.get("modules", {}).items():
        available = _module_symbols(mod)
        for sym in entries.keys():
            # sym may include dotted (Class.method); only check top-level name
            top = sym.split(".")[0]
            if top not in available:
                missing_symbols.append((mod, sym))

    ok = True
    if missing_from_plan:
        ok = False
        print("ERROR: public exports missing from coverage.yaml:", ", ".join(sorted(missing_from_plan)))
    if extra_in_plan:
        ok = False
        print("ERROR: coverage.yaml lists exports not in dashi_core.__all__:", ", ".join(sorted(extra_in_plan)))
    if missing_symbols:
        ok = False
        for mod, sym in missing_symbols:
            print(f"ERROR: coverage.yaml references missing symbol {mod}:{sym}")

    if ok:
        print("coverage check passed")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
