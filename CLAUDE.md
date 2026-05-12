# Working agreements — trellis2-mlx

This file is loaded into every Claude session that touches this repo. It records
non-obvious rules that aren't visible from the code. The authoritative
architecture reference is [`PHASE0_SPEC.md`](PHASE0_SPEC.md) — when this file
and the spec disagree, the spec wins.

## Platform constraints

- **Apple Silicon only.** MLX is the host framework. **No** PyTorch MPS, **no**
  `xformers`, **no** `flash-attn`, **no** `spconv`/`torchsparse`, **no** Triton,
  **no** `nvdiffrast`. Anything CUDA-bound is not in scope.
- **PyTorch is allowed only as a CPU-only numerical reference** under
  `tests/reference/`. It must never appear in the runtime path of the MLX
  pipeline.
- **bf16 for weights and activations.** **fp32 for accumulators** in attention
  and conv kernels (online softmax stats, GEMM accumulators). M4 has native
  bf16 — do not downgrade to fp16 to "save bandwidth"; you lose dynamic range
  and gain nothing.
- **Coordinates are int32** (MLX has no native uint16 dtype).

## Custom-op policy

- **Fine-tuning is in scope.** Every custom Metal op must ship with a backward
  (vjp), even if the immediate use case is inference. The exceptions are
  forward-only-by-nature ops: neighbor table construction, prefix sum,
  mesh extraction, trilinear baking. The forward-only set is enumerated in
  spec §5; do not add to it without an entry in `docs/open-questions-resolved.md`.
- New custom ops live under `trellis2_mlx/metal/` and are bound through
  `trellis2_mlx/metal/ops.py`. For autograd-required ops use the MLX C++
  extension path; for forward-only ops `mx.fast.metal_kernel(...)` is fine.

## Code style

- Type-hint everything. `mypy --strict` is the bar.
  - MLX upstream ships no type stubs, so `mlx.nn.Module` / `mlx.nn.Linear` /
    `mlx.fast.*` all resolve to `Any`. `pyproject.toml` disables
    ``attr-defined / name-defined / misc / no-any-return`` for the
    MLX-bound submodules (``trellis2_mlx.models.*``, ``trellis2_mlx.nn.*``,
    etc.). Revisit when MLX ships stubs. All *non-MLX* code stays strict.
- Docstring every public function/class. Reference the spec section the code
  implements (e.g. `# Implements PHASE0_SPEC.md §5.2`).
- Conventional commit prefixes: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`,
  `perf:`, `chore:`.
- `ruff` and `mypy` configs live in `pyproject.toml`.
- Default to writing no comments. Only add one when the **why** is non-obvious
  (hidden constraints, subtle invariants, references to a spec section).

## Provenance

- **Do NOT fork `shivampkumar/trellis-mac` or copy its code.** This port is
  from-scratch against the Microsoft source. Looking at trellis-mac for ideas
  is fine; copying license-incompatible code is not.
- **Do NOT invent architectural details.** If something is ambiguous in the
  upstream source after reasonable searching, stop and ask. The spec marks
  these with **[VERIFY]**; resolved answers go in
  [`docs/open-questions-resolved.md`](docs/open-questions-resolved.md).
- Upstream Microsoft source is cloned into `reference/microsoft-trellis2/`
  (gitignored). HF weights live in `reference/weights/` (also gitignored).
  Cite upstream sources with `path:line` when resolving open questions.

## Tests

- Unit tests for each custom op compare MLX output against a PyTorch CPU
  reference on small synthetic inputs. Tolerance: `atol=1e-3, rtol=1e-3` for
  bf16 paths, tighter for fp32.
- Tests that require Metal are marked `@pytest.mark.metal`. Tests against the
  PyTorch reference are `@pytest.mark.reference`. Slow tests
  (>5s) get `@pytest.mark.slow`.

## When to stop and ask

- Any **[VERIFY]** item where the upstream source isn't definitive.
- Any design choice not covered by the spec.
- A new top-level dependency not in `pyproject.toml`.
- Anything that would deviate from `PHASE0_SPEC.md`.
