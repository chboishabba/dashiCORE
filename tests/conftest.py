"""
Test configuration to ensure the repository root is importable as a module path.

This makes `import dashi_core` work when running `pytest` from the repo root
without installing the package.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
