# Repository Guidelines

## Project Structure & Module Organization
- Root docs: `README.md` (scope), `CONTRIBUTING.md` (IO contracts), `TESTING.md` (test plan), `SPRINT_*.md` (planning).
- Put reference code in `dashi_core/` with submodules `carrier/`, `kernel/`, `defect/`, `admissibility/`, `mdl/`, `hierarchy/`. Keep GPU/domain backends elsewhere.
- Mirror the testing layout in `TESTING.md` (`tests/carrier/`, `tests/kernel/`, etc.); add `tests/README.md` per area to restate invariants.
- Mock kernels and violation cases live only in tests; production modules stay minimal and invariant-safe.

## Build, Test, and Development Commands
- Create an env: `python -m venv .venv && source .venv/bin/activate`.
- Install deps once packaging lands: `python -m pip install -e .[dev]`.
- Run tests: `python -m pytest tests`; use `pytest -k <pattern>` to focus.
- No build artifacts yet; keep the tree wheel-free until the API is stable.

## Coding Style & Naming Conventions
- Python 3.11+, 4-space indent, explicit type hints; use dataclasses for core objects.
- Names: `Carrier`, `Kernel`, `Defect`, `Admissibility`, `MDL`; functions/modules `lower_snake_case`, classes `PascalCase`.
- Tooling: prefer `ruff` for lint+format and `mypy` for type checks; keep configs small and checked in.
- Purity first: ops are pure and shape-preserving unless documented; kernels never create support unless explicitly allowed.

## Testing Guidelines
- Enforce invariants from `TESTING.md`: shape preservation, ternary carriers, defect ≥ 0, admissibility/MDL invariance.
- Name tests after the invariant (e.g., `test_kernel_shape_preserving`, `test_defect_zero_iff_fixed_point`).
- Use deterministic mock kernels; forbid probabilistic or heuristic behavior.
- Add integration tests chaining Carrier → Kernel → Defect → Admissibility → MDL; include backend equivalence hooks even if only CPU exists.

## Commit & Pull Request Guidelines
- Commits: imperative, scoped (e.g., `Add clamp kernel validation`); mention touched invariants and tests run.
- PRs: brief summary, linked issue, invariants affected, `pytest` output; describe any contract changes.
- Block additions that add domain semantics, performance shortcuts, or silent shape/support changes; document any explicit exception.

## Security & Configuration Tips
- Keep dependencies lean; prefer stdlib/NumPy-level libs only when necessary.
- No hidden normalization, implicit casting, or probabilistic carriers; fail loudly on illegal states.
- Avoid persisting build artifacts; ensure determinism via fixed seeds and deterministic kernels.
