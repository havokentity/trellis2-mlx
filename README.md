# trellis2-mlx

A from-scratch Apple Silicon port of **Microsoft TRELLIS.2** (image-to-3D
generation) on **MLX** with custom **Metal Shading Language** kernels for the
performance-critical sparse operations.

> **Status:** Phase 0 — pre-implementation. Architecture is locked in
> [`PHASE0_SPEC.md`](PHASE0_SPEC.md); code is being scaffolded.

## Why this exists

Microsoft's reference TRELLIS.2 stack is CUDA-bound: it relies on `flash-attn`,
`xformers`, `torchsparse`/`spconv`, `nvdiffrast`, and Triton. None of those run
natively on Apple Silicon. The point of this repo is **native** Apple GPU
performance — not a translation layer on top of PyTorch MPS — by writing the
sparse hot-path kernels (flash-style attention, submanifold sparse conv, neighbor
build, mesh extraction) directly in Metal and binding them through MLX.

## Targets

| Resolution | Output                  | H100 reference | M4 Max realistic target |
|:----------:|:------------------------|:--------------:|:------------------------|
| 512³       | ~2.2K latents, GLB      | 3 s            | 30–45 s                 |
| 1024³      | ~9.6K latents, GLB      | 17 s           | 90–180 s                |
| 1536³      | cascaded                | 60 s           | 5–10 min                |

## Hardware

- Apple Silicon (M-series) with a Metal-capable GPU.
- Minimum 36 GB unified memory recommended for 1024³ at bf16.
- Developed and tested on **M4 Max**.

## Scope

In scope:

- Image-to-3D inference end-to-end (RMBG → DINOv3 → 3 DiT stages → SC-VAE
  decoders → GLB export with PBR materials).
- Fine-tuning the DiT generators (so every custom op ships with a backward).
- bf16 weights and activations; fp32 accumulators.

Out of scope (initial):

- Retraining the SC-VAE from scratch.
- Differentiable rasterization and the `nvdiffrec` split-sum renderer (the GLB
  export does not need them).
- The texturing-only pipeline.

## Repo layout

```
trellis2_mlx/      # MLX modules + Metal kernels (see PHASE0_SPEC.md §6.1)
docs/              # Resolved open questions, weight inventory, design notes
reference/         # Upstream source clone + downloaded HF weights (gitignored)
PHASE0_SPEC.md     # Source-of-truth architecture spec
PHASES.md          # Phased roadmap
CLAUDE.md          # Working agreements for AI-assisted development
```

## Setup (placeholder — code not yet runnable)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## License

MIT. See [LICENSE](LICENSE).

## References

- Paper: Xiang et al., *Native and Compact Structured Latents for 3D Generation*, arXiv 2512.14692
- Upstream: <https://github.com/microsoft/TRELLIS.2>
- Weights: <https://huggingface.co/microsoft/TRELLIS.2-4B>
- MLX: <https://ml-explore.github.io/mlx>

This project is independent of and not affiliated with Microsoft.
