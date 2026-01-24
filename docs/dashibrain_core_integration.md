# dashiBRAIN ↔ dashiCORE Integration Notes

## Support Policy (Locked)
- Use **Option A**: `support = node/channel exists` (constant True for the domain shape).
- Neutral/unknown is a value-level concept, not a support toggle.
- Support must never grow inside kernels; sign carries all dynamics.

## Neutral / Zero Handling
- Legacy `0` values are treated as **neutral masks** external to CORE.
- Adapter maps legacy signed → `Carrier` by:
  - `support = True` everywhere
  - `sign = sign(legacy)` with zeros mapped to `+1` while recording `neutral_mask`
- Export restores zeros via the mask; CORE never sees neutrality encoded as missing support.
- Helper utilities: `dashi_core.adapters.legacy.to_carrier` / `from_carrier` implement this mapping.

## Boundary Split
- `GraphContext` (dashiBRAIN side): adjacency (CSR/COO), hop tables, thresholds, channel metadata.
- `Carrier` (CORE): pure state (`support`, `sign`) with no adjacency.
- Kernels capture `GraphContext` and output validated `Carrier` objects; no graph data enters `Carrier`.

## Parity & Tests to Add
- Adapter roundtrip: `legacy -> Carrier -> legacy` is identity given `neutral_mask`.
- Kernel contract: shape preserved, no support creation, output passes `Carrier.validate`.
- Flow parity: CPU graph flow matches legacy results modulo neutral masking.
- Violation guard: attempts to flip support from False to True must raise.

## GPU Reuse Hooks
- GPU modules stay outside CORE (e.g., `gpu_<role>.py`) and adapt to `Carrier` inputs/outputs.
- Use `gpu_common_methods.py` (`compile_shader`, `find_memory_type`) for Vulkan plumbing; do not import Vulkan into CORE.
- GPU adapters must validate outputs against CORE invariants before returning a `Carrier`.
