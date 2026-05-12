# Weight inventory — microsoft/TRELLIS.2-4B

Generated 2026-05-13 from `reference/weights/ckpts/`.

Source: HF repo `microsoft/TRELLIS.2-4B`, downloaded by snapshot_download.
Files are gitignored under `reference/weights/`; this document is the source-of-truth
mapping that `trellis2_mlx/utils/weight_convert.py` will consume.

A machine-readable copy of the same data is in
[`weight-inventory.json`](weight-inventory.json) (sorted by file, then key).

## Key observations (from the parameter shapes)

These confirm — at the weight level — the spec corrections recorded in
[`open-questions-resolved.md`](open-questions-resolved.md). All examples
below are from `ss_flow_img_dit_1_3B_64_bf16.safetensors` block 0; the
SLAT DiTs have the same per-block layout (only `input_layer.weight` and
`out_layer.weight` differ across stages).

- **Fused QKV projection.** `self_attn.to_qkv.weight: [4608, 1536]` = `[3*1536, 1536]`.
  Q, K, V share a single linear; the binding code must split the output
  evenly along dim 0 (per head: dim 0 reshape to `[3, 12, 128]`).
- **Split Q vs fused KV for cross-attn.** `cross_attn.to_q.weight: [1536, 1536]`,
  `cross_attn.to_kv.weight: [3072, 1024]`. The KV input dim is 1024
  (DINOv3-L feature dim), output is `2 * 1536`. We need a separate
  K/V split for cross-attn but a 3-way split for self-attn.
- **QK-Norm is per-head, per-channel.** `q_rms_norm.gamma: [12, 128]` — one
  scale per head and per channel inside the head, applied before the
  attention dot product.
- **AdaLN-single is shared with per-block learned bias.** Top-level
  `adaLN_modulation.1.{weight,bias}` is the shared `SiLU + Linear(1536 → 9216)`
  (9216 = 6 × 1536); per-block `blocks.k.modulation: [9216]` is the
  learned offset that gets added before chunking into the six
  modulation scalars. This is the PixArt-α `share_mod=True` variant.
- **Only `norm2` carries affine params per block.** `norm1` and `norm3`
  are non-affine (no weight/bias in the safetensors); only the
  cross-attention's LayerNorm has learnable scale/shift. Matches
  `trellis2/modules/sparse/transformer/modulated.py:107`
  (`elementwise_affine=True` for `norm2` only).
- **MLP hidden dim is 8192 exactly** (not 4× = 6144). `mlp.0.weight: [8192, 1536]`,
  `mlp.2.weight: [1536, 8192]`. `mlp_ratio = 5.3334` in the config rounds
  to 8192.
- **SS-DiT latent dimension is 8.** `input_layer.weight: [1536, 8]` and
  `out_layer.weight: [8, 1536]` — confirms stage 1 latent is 8-dim
  (spec §4.4 had 32).
- **Shape decoder output head is 7 channels.** `output_layer.weight: [7, 64]`
  in `shape_dec_*.safetensors`. Slots `0:3` = dual vertex `v`, `3:6` = edge
  flags `δ`, `6:7` = split weight `γ`. Matches §8 Q9.
- **Texture decoder output head is 6 channels.** `output_layer.weight: [6, 64]`
  in `tex_dec_*.safetensors`. Slots `0:3` = base color `c`, `3:4` = metallic `m`,
  `4:5` = roughness `r`, `5:6` = alpha `α`. Matches the `pbr_attr_layout`
  in `trellis2/pipelines/trellis2_image_to_3d.py:73-78`.
- **Early-pruning predictor is one linear per upsample stage.** Each
  `blocks.{i}.{j}.to_subdiv.weight` has shape `[8, channels_at_stage_i]`
  — confirms a *single* `SparseLinear(C, 8)`, no MLP (spec §5.5).
- **Sparse-conv weight layout.** Each `conv1.weight` / `conv2.weight` has
  shape `[out_C, 3, 3, 3, in_C]` — the **kernel comes before the input
  channel dim**, with `(D, H, W) = (3, 3, 3)`. The kernel offset ordering
  encoded by this layout (z-y-x vs x-y-z) is the open item flagged in
  `open-questions-resolved.md`; verify against a known-good neighbor
  lookup during Phase 1 step 5.
- **Total inference parameter budget (single resolution):** about **4.04 B**.
  SS-DiT 1.29 B + one shape SLAT DiT 1.29 B + one tex SLAT DiT 1.29 B +
  shape decoder 0.47 B = 4.04 B. The "4B" in the HF repo name counts the
  inference set, not the 8.12 B total below (which double-counts the
  512/1024 SLAT pairs and the training-only encoders).

## File summary

| File | Role | Params | Size | Dtype |
|---|---|---:|---:|:---:|
| `ss_flow_img_dit_1_3B_64_bf16.safetensors` | Stage 1 — Sparse-Structure DiT | 1,292.2M | 2,584.4 MB | BF16 |
| `slat_flow_img2shape_dit_1_3B_512_bf16.safetensors` | Stage 2 — Shape SLAT DiT (512 variant) | 1,292.3M | 2,584.6 MB | BF16 |
| `slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors` | Stage 2 — Shape SLAT DiT (1024 variant) | 1,292.3M | 2,584.6 MB | BF16 |
| `slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors` | Stage 3 — Texture SLAT DiT (512 variant) | 1,292.3M | 2,584.7 MB | BF16 |
| `slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors` | Stage 3 — Texture SLAT DiT (1024 variant) | 1,292.3M | 2,584.7 MB | BF16 |
| `shape_enc_next_dc_f16c32_fp16.safetensors` | SC-VAE shape encoder | 354.4M | 708.8 MB | F16 |
| `shape_dec_next_dc_f16c32_fp16.safetensors` | SC-VAE shape decoder | 474.2M | 948.5 MB | F16 |
| `tex_enc_next_dc_f16c32_fp16.safetensors` | SC-VAE material encoder | 354.4M | 708.8 MB | F16 |
| `tex_dec_next_dc_f16c32_fp16.safetensors` | SC-VAE material decoder | 474.2M | 948.5 MB | F16 |
| **TOTAL** | | **8,118.5M (8.12B)** | | |

## `ss_flow_img_dit_1_3B_64_bf16.safetensors`

**Role:** Stage 1 — Sparse-Structure DiT

Dense 16³ × 8ch flow model. Produces stage-1 SS latents which the legacy TRELLIS-1 dense SS-VAE decoder upsamples to 64³ binary occupancy.

Total: **640 parameters**, 1292.18M elements.

Top-level prefixes: `adaLN_modulation` (2), `blocks` (630), `input_layer` (2), `out_layer` (2), `t_embedder` (4)

| # | Parameter | Shape | Dtype | Elements |
|---:|---|---|:---:|---:|
| 1 | `adaLN_modulation.1.bias` | [9216] | BF16 | 9,216 |
| 2 | `adaLN_modulation.1.weight` | [9216, 1536] | BF16 | 14,155,776 |
| 3 | `blocks.0.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 4 | `blocks.0.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 5 | `blocks.0.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 6 | `blocks.0.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 7 | `blocks.0.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 8 | `blocks.0.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 9 | `blocks.0.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 10 | `blocks.0.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 11 | `blocks.0.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 12 | `blocks.0.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 13 | `blocks.0.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 14 | `blocks.0.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 15 | `blocks.0.modulation` | [9216] | BF16 | 9,216 |
| 16 | `blocks.0.norm2.bias` | [1536] | BF16 | 1,536 |
| 17 | `blocks.0.norm2.weight` | [1536] | BF16 | 1,536 |
| 18 | `blocks.0.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 19 | `blocks.0.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 20 | `blocks.0.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 21 | `blocks.0.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 22 | `blocks.0.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 23 | `blocks.0.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 24 | `blocks.1.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 25 | `blocks.1.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 26 | `blocks.1.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 27 | `blocks.1.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 28 | `blocks.1.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 29 | `blocks.1.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 30 | `blocks.1.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 31 | `blocks.1.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 32 | `blocks.1.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 33 | `blocks.1.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 34 | `blocks.1.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 35 | `blocks.1.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 36 | `blocks.1.modulation` | [9216] | BF16 | 9,216 |
| 37 | `blocks.1.norm2.bias` | [1536] | BF16 | 1,536 |
| 38 | `blocks.1.norm2.weight` | [1536] | BF16 | 1,536 |
| 39 | `blocks.1.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 40 | `blocks.1.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 41 | `blocks.1.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 42 | `blocks.1.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 43 | `blocks.1.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 44 | `blocks.1.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 45 | `blocks.10.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 46 | `blocks.10.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 47 | `blocks.10.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 48 | `blocks.10.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 49 | `blocks.10.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 50 | `blocks.10.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 51 | `blocks.10.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 52 | `blocks.10.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 53 | `blocks.10.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 54 | `blocks.10.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 55 | `blocks.10.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 56 | `blocks.10.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 57 | `blocks.10.modulation` | [9216] | BF16 | 9,216 |
| 58 | `blocks.10.norm2.bias` | [1536] | BF16 | 1,536 |
| 59 | `blocks.10.norm2.weight` | [1536] | BF16 | 1,536 |
| 60 | `blocks.10.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 61 | `blocks.10.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 62 | `blocks.10.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 63 | `blocks.10.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 64 | `blocks.10.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 65 | `blocks.10.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 66 | `blocks.11.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 67 | `blocks.11.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 68 | `blocks.11.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 69 | `blocks.11.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 70 | `blocks.11.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 71 | `blocks.11.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 72 | `blocks.11.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 73 | `blocks.11.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 74 | `blocks.11.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 75 | `blocks.11.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 76 | `blocks.11.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 77 | `blocks.11.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 78 | `blocks.11.modulation` | [9216] | BF16 | 9,216 |
| 79 | `blocks.11.norm2.bias` | [1536] | BF16 | 1,536 |
| 80 | `blocks.11.norm2.weight` | [1536] | BF16 | 1,536 |
| 81 | `blocks.11.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 82 | `blocks.11.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 83 | `blocks.11.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 84 | `blocks.11.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 85 | `blocks.11.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 86 | `blocks.11.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 87 | `blocks.12.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 88 | `blocks.12.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 89 | `blocks.12.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 90 | `blocks.12.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 91 | `blocks.12.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 92 | `blocks.12.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 93 | `blocks.12.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 94 | `blocks.12.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 95 | `blocks.12.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 96 | `blocks.12.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 97 | `blocks.12.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 98 | `blocks.12.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 99 | `blocks.12.modulation` | [9216] | BF16 | 9,216 |
| 100 | `blocks.12.norm2.bias` | [1536] | BF16 | 1,536 |
| 101 | `blocks.12.norm2.weight` | [1536] | BF16 | 1,536 |
| 102 | `blocks.12.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 103 | `blocks.12.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 104 | `blocks.12.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 105 | `blocks.12.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 106 | `blocks.12.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 107 | `blocks.12.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 108 | `blocks.13.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 109 | `blocks.13.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 110 | `blocks.13.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 111 | `blocks.13.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 112 | `blocks.13.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 113 | `blocks.13.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 114 | `blocks.13.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 115 | `blocks.13.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 116 | `blocks.13.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 117 | `blocks.13.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 118 | `blocks.13.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 119 | `blocks.13.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 120 | `blocks.13.modulation` | [9216] | BF16 | 9,216 |
| 121 | `blocks.13.norm2.bias` | [1536] | BF16 | 1,536 |
| 122 | `blocks.13.norm2.weight` | [1536] | BF16 | 1,536 |
| 123 | `blocks.13.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 124 | `blocks.13.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 125 | `blocks.13.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 126 | `blocks.13.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 127 | `blocks.13.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 128 | `blocks.13.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 129 | `blocks.14.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 130 | `blocks.14.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 131 | `blocks.14.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 132 | `blocks.14.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 133 | `blocks.14.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 134 | `blocks.14.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 135 | `blocks.14.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 136 | `blocks.14.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 137 | `blocks.14.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 138 | `blocks.14.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 139 | `blocks.14.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 140 | `blocks.14.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 141 | `blocks.14.modulation` | [9216] | BF16 | 9,216 |
| 142 | `blocks.14.norm2.bias` | [1536] | BF16 | 1,536 |
| 143 | `blocks.14.norm2.weight` | [1536] | BF16 | 1,536 |
| 144 | `blocks.14.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 145 | `blocks.14.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 146 | `blocks.14.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 147 | `blocks.14.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 148 | `blocks.14.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 149 | `blocks.14.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 150 | `blocks.15.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 151 | `blocks.15.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 152 | `blocks.15.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 153 | `blocks.15.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 154 | `blocks.15.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 155 | `blocks.15.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 156 | `blocks.15.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 157 | `blocks.15.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 158 | `blocks.15.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 159 | `blocks.15.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 160 | `blocks.15.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 161 | `blocks.15.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 162 | `blocks.15.modulation` | [9216] | BF16 | 9,216 |
| 163 | `blocks.15.norm2.bias` | [1536] | BF16 | 1,536 |
| 164 | `blocks.15.norm2.weight` | [1536] | BF16 | 1,536 |
| 165 | `blocks.15.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 166 | `blocks.15.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 167 | `blocks.15.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 168 | `blocks.15.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 169 | `blocks.15.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 170 | `blocks.15.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 171 | `blocks.16.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 172 | `blocks.16.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 173 | `blocks.16.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 174 | `blocks.16.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 175 | `blocks.16.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 176 | `blocks.16.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 177 | `blocks.16.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 178 | `blocks.16.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 179 | `blocks.16.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 180 | `blocks.16.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 181 | `blocks.16.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 182 | `blocks.16.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 183 | `blocks.16.modulation` | [9216] | BF16 | 9,216 |
| 184 | `blocks.16.norm2.bias` | [1536] | BF16 | 1,536 |
| 185 | `blocks.16.norm2.weight` | [1536] | BF16 | 1,536 |
| 186 | `blocks.16.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 187 | `blocks.16.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 188 | `blocks.16.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 189 | `blocks.16.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 190 | `blocks.16.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 191 | `blocks.16.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 192 | `blocks.17.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 193 | `blocks.17.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 194 | `blocks.17.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 195 | `blocks.17.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 196 | `blocks.17.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 197 | `blocks.17.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 198 | `blocks.17.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 199 | `blocks.17.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 200 | `blocks.17.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 201 | `blocks.17.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 202 | `blocks.17.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 203 | `blocks.17.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 204 | `blocks.17.modulation` | [9216] | BF16 | 9,216 |
| 205 | `blocks.17.norm2.bias` | [1536] | BF16 | 1,536 |
| 206 | `blocks.17.norm2.weight` | [1536] | BF16 | 1,536 |
| 207 | `blocks.17.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 208 | `blocks.17.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 209 | `blocks.17.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 210 | `blocks.17.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 211 | `blocks.17.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 212 | `blocks.17.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 213 | `blocks.18.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 214 | `blocks.18.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 215 | `blocks.18.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 216 | `blocks.18.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 217 | `blocks.18.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 218 | `blocks.18.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 219 | `blocks.18.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 220 | `blocks.18.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 221 | `blocks.18.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 222 | `blocks.18.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 223 | `blocks.18.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 224 | `blocks.18.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 225 | `blocks.18.modulation` | [9216] | BF16 | 9,216 |
| 226 | `blocks.18.norm2.bias` | [1536] | BF16 | 1,536 |
| 227 | `blocks.18.norm2.weight` | [1536] | BF16 | 1,536 |
| 228 | `blocks.18.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 229 | `blocks.18.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 230 | `blocks.18.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 231 | `blocks.18.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 232 | `blocks.18.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 233 | `blocks.18.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 234 | `blocks.19.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 235 | `blocks.19.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 236 | `blocks.19.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 237 | `blocks.19.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 238 | `blocks.19.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 239 | `blocks.19.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 240 | `blocks.19.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 241 | `blocks.19.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 242 | `blocks.19.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 243 | `blocks.19.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 244 | `blocks.19.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 245 | `blocks.19.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 246 | `blocks.19.modulation` | [9216] | BF16 | 9,216 |
| 247 | `blocks.19.norm2.bias` | [1536] | BF16 | 1,536 |
| 248 | `blocks.19.norm2.weight` | [1536] | BF16 | 1,536 |
| 249 | `blocks.19.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 250 | `blocks.19.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 251 | `blocks.19.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 252 | `blocks.19.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 253 | `blocks.19.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 254 | `blocks.19.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 255 | `blocks.2.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 256 | `blocks.2.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 257 | `blocks.2.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 258 | `blocks.2.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 259 | `blocks.2.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 260 | `blocks.2.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 261 | `blocks.2.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 262 | `blocks.2.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 263 | `blocks.2.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 264 | `blocks.2.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 265 | `blocks.2.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 266 | `blocks.2.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 267 | `blocks.2.modulation` | [9216] | BF16 | 9,216 |
| 268 | `blocks.2.norm2.bias` | [1536] | BF16 | 1,536 |
| 269 | `blocks.2.norm2.weight` | [1536] | BF16 | 1,536 |
| 270 | `blocks.2.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 271 | `blocks.2.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 272 | `blocks.2.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 273 | `blocks.2.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 274 | `blocks.2.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 275 | `blocks.2.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 276 | `blocks.20.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 277 | `blocks.20.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 278 | `blocks.20.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 279 | `blocks.20.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 280 | `blocks.20.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 281 | `blocks.20.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 282 | `blocks.20.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 283 | `blocks.20.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 284 | `blocks.20.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 285 | `blocks.20.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 286 | `blocks.20.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 287 | `blocks.20.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 288 | `blocks.20.modulation` | [9216] | BF16 | 9,216 |
| 289 | `blocks.20.norm2.bias` | [1536] | BF16 | 1,536 |
| 290 | `blocks.20.norm2.weight` | [1536] | BF16 | 1,536 |
| 291 | `blocks.20.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 292 | `blocks.20.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 293 | `blocks.20.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 294 | `blocks.20.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 295 | `blocks.20.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 296 | `blocks.20.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 297 | `blocks.21.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 298 | `blocks.21.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 299 | `blocks.21.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 300 | `blocks.21.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 301 | `blocks.21.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 302 | `blocks.21.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 303 | `blocks.21.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 304 | `blocks.21.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 305 | `blocks.21.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 306 | `blocks.21.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 307 | `blocks.21.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 308 | `blocks.21.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 309 | `blocks.21.modulation` | [9216] | BF16 | 9,216 |
| 310 | `blocks.21.norm2.bias` | [1536] | BF16 | 1,536 |
| 311 | `blocks.21.norm2.weight` | [1536] | BF16 | 1,536 |
| 312 | `blocks.21.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 313 | `blocks.21.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 314 | `blocks.21.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 315 | `blocks.21.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 316 | `blocks.21.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 317 | `blocks.21.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 318 | `blocks.22.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 319 | `blocks.22.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 320 | `blocks.22.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 321 | `blocks.22.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 322 | `blocks.22.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 323 | `blocks.22.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 324 | `blocks.22.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 325 | `blocks.22.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 326 | `blocks.22.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 327 | `blocks.22.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 328 | `blocks.22.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 329 | `blocks.22.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 330 | `blocks.22.modulation` | [9216] | BF16 | 9,216 |
| 331 | `blocks.22.norm2.bias` | [1536] | BF16 | 1,536 |
| 332 | `blocks.22.norm2.weight` | [1536] | BF16 | 1,536 |
| 333 | `blocks.22.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 334 | `blocks.22.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 335 | `blocks.22.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 336 | `blocks.22.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 337 | `blocks.22.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 338 | `blocks.22.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 339 | `blocks.23.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 340 | `blocks.23.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 341 | `blocks.23.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 342 | `blocks.23.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 343 | `blocks.23.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 344 | `blocks.23.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 345 | `blocks.23.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 346 | `blocks.23.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 347 | `blocks.23.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 348 | `blocks.23.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 349 | `blocks.23.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 350 | `blocks.23.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 351 | `blocks.23.modulation` | [9216] | BF16 | 9,216 |
| 352 | `blocks.23.norm2.bias` | [1536] | BF16 | 1,536 |
| 353 | `blocks.23.norm2.weight` | [1536] | BF16 | 1,536 |
| 354 | `blocks.23.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 355 | `blocks.23.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 356 | `blocks.23.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 357 | `blocks.23.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 358 | `blocks.23.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 359 | `blocks.23.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 360 | `blocks.24.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 361 | `blocks.24.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 362 | `blocks.24.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 363 | `blocks.24.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 364 | `blocks.24.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 365 | `blocks.24.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 366 | `blocks.24.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 367 | `blocks.24.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 368 | `blocks.24.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 369 | `blocks.24.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 370 | `blocks.24.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 371 | `blocks.24.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 372 | `blocks.24.modulation` | [9216] | BF16 | 9,216 |
| 373 | `blocks.24.norm2.bias` | [1536] | BF16 | 1,536 |
| 374 | `blocks.24.norm2.weight` | [1536] | BF16 | 1,536 |
| 375 | `blocks.24.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 376 | `blocks.24.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 377 | `blocks.24.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 378 | `blocks.24.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 379 | `blocks.24.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 380 | `blocks.24.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 381 | `blocks.25.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 382 | `blocks.25.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 383 | `blocks.25.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 384 | `blocks.25.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 385 | `blocks.25.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 386 | `blocks.25.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 387 | `blocks.25.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 388 | `blocks.25.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 389 | `blocks.25.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 390 | `blocks.25.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 391 | `blocks.25.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 392 | `blocks.25.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 393 | `blocks.25.modulation` | [9216] | BF16 | 9,216 |
| 394 | `blocks.25.norm2.bias` | [1536] | BF16 | 1,536 |
| 395 | `blocks.25.norm2.weight` | [1536] | BF16 | 1,536 |
| 396 | `blocks.25.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 397 | `blocks.25.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 398 | `blocks.25.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 399 | `blocks.25.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 400 | `blocks.25.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 401 | `blocks.25.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 402 | `blocks.26.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 403 | `blocks.26.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 404 | `blocks.26.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 405 | `blocks.26.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 406 | `blocks.26.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 407 | `blocks.26.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 408 | `blocks.26.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 409 | `blocks.26.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 410 | `blocks.26.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 411 | `blocks.26.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 412 | `blocks.26.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 413 | `blocks.26.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 414 | `blocks.26.modulation` | [9216] | BF16 | 9,216 |
| 415 | `blocks.26.norm2.bias` | [1536] | BF16 | 1,536 |
| 416 | `blocks.26.norm2.weight` | [1536] | BF16 | 1,536 |
| 417 | `blocks.26.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 418 | `blocks.26.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 419 | `blocks.26.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 420 | `blocks.26.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 421 | `blocks.26.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 422 | `blocks.26.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 423 | `blocks.27.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 424 | `blocks.27.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 425 | `blocks.27.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 426 | `blocks.27.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 427 | `blocks.27.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 428 | `blocks.27.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 429 | `blocks.27.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 430 | `blocks.27.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 431 | `blocks.27.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 432 | `blocks.27.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 433 | `blocks.27.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 434 | `blocks.27.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 435 | `blocks.27.modulation` | [9216] | BF16 | 9,216 |
| 436 | `blocks.27.norm2.bias` | [1536] | BF16 | 1,536 |
| 437 | `blocks.27.norm2.weight` | [1536] | BF16 | 1,536 |
| 438 | `blocks.27.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 439 | `blocks.27.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 440 | `blocks.27.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 441 | `blocks.27.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 442 | `blocks.27.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 443 | `blocks.27.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 444 | `blocks.28.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 445 | `blocks.28.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 446 | `blocks.28.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 447 | `blocks.28.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 448 | `blocks.28.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 449 | `blocks.28.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 450 | `blocks.28.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 451 | `blocks.28.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 452 | `blocks.28.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 453 | `blocks.28.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 454 | `blocks.28.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 455 | `blocks.28.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 456 | `blocks.28.modulation` | [9216] | BF16 | 9,216 |
| 457 | `blocks.28.norm2.bias` | [1536] | BF16 | 1,536 |
| 458 | `blocks.28.norm2.weight` | [1536] | BF16 | 1,536 |
| 459 | `blocks.28.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 460 | `blocks.28.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 461 | `blocks.28.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 462 | `blocks.28.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 463 | `blocks.28.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 464 | `blocks.28.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 465 | `blocks.29.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 466 | `blocks.29.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 467 | `blocks.29.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 468 | `blocks.29.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 469 | `blocks.29.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 470 | `blocks.29.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 471 | `blocks.29.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 472 | `blocks.29.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 473 | `blocks.29.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 474 | `blocks.29.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 475 | `blocks.29.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 476 | `blocks.29.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 477 | `blocks.29.modulation` | [9216] | BF16 | 9,216 |
| 478 | `blocks.29.norm2.bias` | [1536] | BF16 | 1,536 |
| 479 | `blocks.29.norm2.weight` | [1536] | BF16 | 1,536 |
| 480 | `blocks.29.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 481 | `blocks.29.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 482 | `blocks.29.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 483 | `blocks.29.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 484 | `blocks.29.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 485 | `blocks.29.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 486 | `blocks.3.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 487 | `blocks.3.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 488 | `blocks.3.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 489 | `blocks.3.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 490 | `blocks.3.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 491 | `blocks.3.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 492 | `blocks.3.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 493 | `blocks.3.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 494 | `blocks.3.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 495 | `blocks.3.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 496 | `blocks.3.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 497 | `blocks.3.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 498 | `blocks.3.modulation` | [9216] | BF16 | 9,216 |
| 499 | `blocks.3.norm2.bias` | [1536] | BF16 | 1,536 |
| 500 | `blocks.3.norm2.weight` | [1536] | BF16 | 1,536 |
| 501 | `blocks.3.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 502 | `blocks.3.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 503 | `blocks.3.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 504 | `blocks.3.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 505 | `blocks.3.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 506 | `blocks.3.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 507 | `blocks.4.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 508 | `blocks.4.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 509 | `blocks.4.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 510 | `blocks.4.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 511 | `blocks.4.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 512 | `blocks.4.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 513 | `blocks.4.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 514 | `blocks.4.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 515 | `blocks.4.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 516 | `blocks.4.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 517 | `blocks.4.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 518 | `blocks.4.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 519 | `blocks.4.modulation` | [9216] | BF16 | 9,216 |
| 520 | `blocks.4.norm2.bias` | [1536] | BF16 | 1,536 |
| 521 | `blocks.4.norm2.weight` | [1536] | BF16 | 1,536 |
| 522 | `blocks.4.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 523 | `blocks.4.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 524 | `blocks.4.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 525 | `blocks.4.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 526 | `blocks.4.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 527 | `blocks.4.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 528 | `blocks.5.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 529 | `blocks.5.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 530 | `blocks.5.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 531 | `blocks.5.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 532 | `blocks.5.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 533 | `blocks.5.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 534 | `blocks.5.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 535 | `blocks.5.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 536 | `blocks.5.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 537 | `blocks.5.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 538 | `blocks.5.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 539 | `blocks.5.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 540 | `blocks.5.modulation` | [9216] | BF16 | 9,216 |
| 541 | `blocks.5.norm2.bias` | [1536] | BF16 | 1,536 |
| 542 | `blocks.5.norm2.weight` | [1536] | BF16 | 1,536 |
| 543 | `blocks.5.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 544 | `blocks.5.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 545 | `blocks.5.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 546 | `blocks.5.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 547 | `blocks.5.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 548 | `blocks.5.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 549 | `blocks.6.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 550 | `blocks.6.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 551 | `blocks.6.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 552 | `blocks.6.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 553 | `blocks.6.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 554 | `blocks.6.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 555 | `blocks.6.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 556 | `blocks.6.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 557 | `blocks.6.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 558 | `blocks.6.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 559 | `blocks.6.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 560 | `blocks.6.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 561 | `blocks.6.modulation` | [9216] | BF16 | 9,216 |
| 562 | `blocks.6.norm2.bias` | [1536] | BF16 | 1,536 |
| 563 | `blocks.6.norm2.weight` | [1536] | BF16 | 1,536 |
| 564 | `blocks.6.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 565 | `blocks.6.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 566 | `blocks.6.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 567 | `blocks.6.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 568 | `blocks.6.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 569 | `blocks.6.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 570 | `blocks.7.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 571 | `blocks.7.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 572 | `blocks.7.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 573 | `blocks.7.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 574 | `blocks.7.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 575 | `blocks.7.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 576 | `blocks.7.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 577 | `blocks.7.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 578 | `blocks.7.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 579 | `blocks.7.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 580 | `blocks.7.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 581 | `blocks.7.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 582 | `blocks.7.modulation` | [9216] | BF16 | 9,216 |
| 583 | `blocks.7.norm2.bias` | [1536] | BF16 | 1,536 |
| 584 | `blocks.7.norm2.weight` | [1536] | BF16 | 1,536 |
| 585 | `blocks.7.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 586 | `blocks.7.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 587 | `blocks.7.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 588 | `blocks.7.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 589 | `blocks.7.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 590 | `blocks.7.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 591 | `blocks.8.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 592 | `blocks.8.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 593 | `blocks.8.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 594 | `blocks.8.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 595 | `blocks.8.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 596 | `blocks.8.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 597 | `blocks.8.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 598 | `blocks.8.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 599 | `blocks.8.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 600 | `blocks.8.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 601 | `blocks.8.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 602 | `blocks.8.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 603 | `blocks.8.modulation` | [9216] | BF16 | 9,216 |
| 604 | `blocks.8.norm2.bias` | [1536] | BF16 | 1,536 |
| 605 | `blocks.8.norm2.weight` | [1536] | BF16 | 1,536 |
| 606 | `blocks.8.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 607 | `blocks.8.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 608 | `blocks.8.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 609 | `blocks.8.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 610 | `blocks.8.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 611 | `blocks.8.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 612 | `blocks.9.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 613 | `blocks.9.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 614 | `blocks.9.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 615 | `blocks.9.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 616 | `blocks.9.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 617 | `blocks.9.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 618 | `blocks.9.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 619 | `blocks.9.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 620 | `blocks.9.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 621 | `blocks.9.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 622 | `blocks.9.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 623 | `blocks.9.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 624 | `blocks.9.modulation` | [9216] | BF16 | 9,216 |
| 625 | `blocks.9.norm2.bias` | [1536] | BF16 | 1,536 |
| 626 | `blocks.9.norm2.weight` | [1536] | BF16 | 1,536 |
| 627 | `blocks.9.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 628 | `blocks.9.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 629 | `blocks.9.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 630 | `blocks.9.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 631 | `blocks.9.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 632 | `blocks.9.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 633 | `input_layer.bias` | [1536] | BF16 | 1,536 |
| 634 | `input_layer.weight` | [1536, 8] | BF16 | 12,288 |
| 635 | `out_layer.bias` | [8] | BF16 | 8 |
| 636 | `out_layer.weight` | [8, 1536] | BF16 | 12,288 |
| 637 | `t_embedder.mlp.0.bias` | [1536] | BF16 | 1,536 |
| 638 | `t_embedder.mlp.0.weight` | [1536, 256] | BF16 | 393,216 |
| 639 | `t_embedder.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 640 | `t_embedder.mlp.2.weight` | [1536, 1536] | BF16 | 2,359,296 |

## `slat_flow_img2shape_dit_1_3B_512_bf16.safetensors`

**Role:** Stage 2 — Shape SLAT DiT (512 variant)

Sparse 32³ × 32ch geometry flow model used for 512³ outputs and as the LR step in the 1024_cascade / 1536_cascade pipelines.

Total: **640 parameters**, 1292.25M elements.

Top-level prefixes: `adaLN_modulation` (2), `blocks` (630), `input_layer` (2), `out_layer` (2), `t_embedder` (4)

| # | Parameter | Shape | Dtype | Elements |
|---:|---|---|:---:|---:|
| 1 | `adaLN_modulation.1.bias` | [9216] | BF16 | 9,216 |
| 2 | `adaLN_modulation.1.weight` | [9216, 1536] | BF16 | 14,155,776 |
| 3 | `blocks.0.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 4 | `blocks.0.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 5 | `blocks.0.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 6 | `blocks.0.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 7 | `blocks.0.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 8 | `blocks.0.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 9 | `blocks.0.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 10 | `blocks.0.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 11 | `blocks.0.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 12 | `blocks.0.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 13 | `blocks.0.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 14 | `blocks.0.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 15 | `blocks.0.modulation` | [9216] | BF16 | 9,216 |
| 16 | `blocks.0.norm2.bias` | [1536] | BF16 | 1,536 |
| 17 | `blocks.0.norm2.weight` | [1536] | BF16 | 1,536 |
| 18 | `blocks.0.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 19 | `blocks.0.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 20 | `blocks.0.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 21 | `blocks.0.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 22 | `blocks.0.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 23 | `blocks.0.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 24 | `blocks.1.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 25 | `blocks.1.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 26 | `blocks.1.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 27 | `blocks.1.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 28 | `blocks.1.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 29 | `blocks.1.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 30 | `blocks.1.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 31 | `blocks.1.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 32 | `blocks.1.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 33 | `blocks.1.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 34 | `blocks.1.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 35 | `blocks.1.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 36 | `blocks.1.modulation` | [9216] | BF16 | 9,216 |
| 37 | `blocks.1.norm2.bias` | [1536] | BF16 | 1,536 |
| 38 | `blocks.1.norm2.weight` | [1536] | BF16 | 1,536 |
| 39 | `blocks.1.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 40 | `blocks.1.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 41 | `blocks.1.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 42 | `blocks.1.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 43 | `blocks.1.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 44 | `blocks.1.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 45 | `blocks.10.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 46 | `blocks.10.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 47 | `blocks.10.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 48 | `blocks.10.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 49 | `blocks.10.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 50 | `blocks.10.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 51 | `blocks.10.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 52 | `blocks.10.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 53 | `blocks.10.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 54 | `blocks.10.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 55 | `blocks.10.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 56 | `blocks.10.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 57 | `blocks.10.modulation` | [9216] | BF16 | 9,216 |
| 58 | `blocks.10.norm2.bias` | [1536] | BF16 | 1,536 |
| 59 | `blocks.10.norm2.weight` | [1536] | BF16 | 1,536 |
| 60 | `blocks.10.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 61 | `blocks.10.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 62 | `blocks.10.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 63 | `blocks.10.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 64 | `blocks.10.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 65 | `blocks.10.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 66 | `blocks.11.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 67 | `blocks.11.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 68 | `blocks.11.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 69 | `blocks.11.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 70 | `blocks.11.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 71 | `blocks.11.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 72 | `blocks.11.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 73 | `blocks.11.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 74 | `blocks.11.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 75 | `blocks.11.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 76 | `blocks.11.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 77 | `blocks.11.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 78 | `blocks.11.modulation` | [9216] | BF16 | 9,216 |
| 79 | `blocks.11.norm2.bias` | [1536] | BF16 | 1,536 |
| 80 | `blocks.11.norm2.weight` | [1536] | BF16 | 1,536 |
| 81 | `blocks.11.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 82 | `blocks.11.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 83 | `blocks.11.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 84 | `blocks.11.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 85 | `blocks.11.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 86 | `blocks.11.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 87 | `blocks.12.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 88 | `blocks.12.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 89 | `blocks.12.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 90 | `blocks.12.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 91 | `blocks.12.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 92 | `blocks.12.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 93 | `blocks.12.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 94 | `blocks.12.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 95 | `blocks.12.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 96 | `blocks.12.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 97 | `blocks.12.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 98 | `blocks.12.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 99 | `blocks.12.modulation` | [9216] | BF16 | 9,216 |
| 100 | `blocks.12.norm2.bias` | [1536] | BF16 | 1,536 |
| 101 | `blocks.12.norm2.weight` | [1536] | BF16 | 1,536 |
| 102 | `blocks.12.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 103 | `blocks.12.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 104 | `blocks.12.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 105 | `blocks.12.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 106 | `blocks.12.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 107 | `blocks.12.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 108 | `blocks.13.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 109 | `blocks.13.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 110 | `blocks.13.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 111 | `blocks.13.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 112 | `blocks.13.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 113 | `blocks.13.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 114 | `blocks.13.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 115 | `blocks.13.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 116 | `blocks.13.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 117 | `blocks.13.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 118 | `blocks.13.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 119 | `blocks.13.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 120 | `blocks.13.modulation` | [9216] | BF16 | 9,216 |
| 121 | `blocks.13.norm2.bias` | [1536] | BF16 | 1,536 |
| 122 | `blocks.13.norm2.weight` | [1536] | BF16 | 1,536 |
| 123 | `blocks.13.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 124 | `blocks.13.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 125 | `blocks.13.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 126 | `blocks.13.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 127 | `blocks.13.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 128 | `blocks.13.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 129 | `blocks.14.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 130 | `blocks.14.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 131 | `blocks.14.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 132 | `blocks.14.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 133 | `blocks.14.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 134 | `blocks.14.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 135 | `blocks.14.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 136 | `blocks.14.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 137 | `blocks.14.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 138 | `blocks.14.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 139 | `blocks.14.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 140 | `blocks.14.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 141 | `blocks.14.modulation` | [9216] | BF16 | 9,216 |
| 142 | `blocks.14.norm2.bias` | [1536] | BF16 | 1,536 |
| 143 | `blocks.14.norm2.weight` | [1536] | BF16 | 1,536 |
| 144 | `blocks.14.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 145 | `blocks.14.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 146 | `blocks.14.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 147 | `blocks.14.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 148 | `blocks.14.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 149 | `blocks.14.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 150 | `blocks.15.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 151 | `blocks.15.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 152 | `blocks.15.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 153 | `blocks.15.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 154 | `blocks.15.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 155 | `blocks.15.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 156 | `blocks.15.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 157 | `blocks.15.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 158 | `blocks.15.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 159 | `blocks.15.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 160 | `blocks.15.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 161 | `blocks.15.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 162 | `blocks.15.modulation` | [9216] | BF16 | 9,216 |
| 163 | `blocks.15.norm2.bias` | [1536] | BF16 | 1,536 |
| 164 | `blocks.15.norm2.weight` | [1536] | BF16 | 1,536 |
| 165 | `blocks.15.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 166 | `blocks.15.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 167 | `blocks.15.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 168 | `blocks.15.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 169 | `blocks.15.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 170 | `blocks.15.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 171 | `blocks.16.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 172 | `blocks.16.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 173 | `blocks.16.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 174 | `blocks.16.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 175 | `blocks.16.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 176 | `blocks.16.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 177 | `blocks.16.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 178 | `blocks.16.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 179 | `blocks.16.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 180 | `blocks.16.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 181 | `blocks.16.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 182 | `blocks.16.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 183 | `blocks.16.modulation` | [9216] | BF16 | 9,216 |
| 184 | `blocks.16.norm2.bias` | [1536] | BF16 | 1,536 |
| 185 | `blocks.16.norm2.weight` | [1536] | BF16 | 1,536 |
| 186 | `blocks.16.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 187 | `blocks.16.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 188 | `blocks.16.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 189 | `blocks.16.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 190 | `blocks.16.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 191 | `blocks.16.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 192 | `blocks.17.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 193 | `blocks.17.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 194 | `blocks.17.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 195 | `blocks.17.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 196 | `blocks.17.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 197 | `blocks.17.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 198 | `blocks.17.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 199 | `blocks.17.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 200 | `blocks.17.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 201 | `blocks.17.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 202 | `blocks.17.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 203 | `blocks.17.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 204 | `blocks.17.modulation` | [9216] | BF16 | 9,216 |
| 205 | `blocks.17.norm2.bias` | [1536] | BF16 | 1,536 |
| 206 | `blocks.17.norm2.weight` | [1536] | BF16 | 1,536 |
| 207 | `blocks.17.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 208 | `blocks.17.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 209 | `blocks.17.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 210 | `blocks.17.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 211 | `blocks.17.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 212 | `blocks.17.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 213 | `blocks.18.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 214 | `blocks.18.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 215 | `blocks.18.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 216 | `blocks.18.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 217 | `blocks.18.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 218 | `blocks.18.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 219 | `blocks.18.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 220 | `blocks.18.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 221 | `blocks.18.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 222 | `blocks.18.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 223 | `blocks.18.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 224 | `blocks.18.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 225 | `blocks.18.modulation` | [9216] | BF16 | 9,216 |
| 226 | `blocks.18.norm2.bias` | [1536] | BF16 | 1,536 |
| 227 | `blocks.18.norm2.weight` | [1536] | BF16 | 1,536 |
| 228 | `blocks.18.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 229 | `blocks.18.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 230 | `blocks.18.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 231 | `blocks.18.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 232 | `blocks.18.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 233 | `blocks.18.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 234 | `blocks.19.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 235 | `blocks.19.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 236 | `blocks.19.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 237 | `blocks.19.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 238 | `blocks.19.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 239 | `blocks.19.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 240 | `blocks.19.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 241 | `blocks.19.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 242 | `blocks.19.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 243 | `blocks.19.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 244 | `blocks.19.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 245 | `blocks.19.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 246 | `blocks.19.modulation` | [9216] | BF16 | 9,216 |
| 247 | `blocks.19.norm2.bias` | [1536] | BF16 | 1,536 |
| 248 | `blocks.19.norm2.weight` | [1536] | BF16 | 1,536 |
| 249 | `blocks.19.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 250 | `blocks.19.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 251 | `blocks.19.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 252 | `blocks.19.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 253 | `blocks.19.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 254 | `blocks.19.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 255 | `blocks.2.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 256 | `blocks.2.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 257 | `blocks.2.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 258 | `blocks.2.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 259 | `blocks.2.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 260 | `blocks.2.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 261 | `blocks.2.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 262 | `blocks.2.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 263 | `blocks.2.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 264 | `blocks.2.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 265 | `blocks.2.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 266 | `blocks.2.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 267 | `blocks.2.modulation` | [9216] | BF16 | 9,216 |
| 268 | `blocks.2.norm2.bias` | [1536] | BF16 | 1,536 |
| 269 | `blocks.2.norm2.weight` | [1536] | BF16 | 1,536 |
| 270 | `blocks.2.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 271 | `blocks.2.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 272 | `blocks.2.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 273 | `blocks.2.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 274 | `blocks.2.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 275 | `blocks.2.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 276 | `blocks.20.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 277 | `blocks.20.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 278 | `blocks.20.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 279 | `blocks.20.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 280 | `blocks.20.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 281 | `blocks.20.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 282 | `blocks.20.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 283 | `blocks.20.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 284 | `blocks.20.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 285 | `blocks.20.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 286 | `blocks.20.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 287 | `blocks.20.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 288 | `blocks.20.modulation` | [9216] | BF16 | 9,216 |
| 289 | `blocks.20.norm2.bias` | [1536] | BF16 | 1,536 |
| 290 | `blocks.20.norm2.weight` | [1536] | BF16 | 1,536 |
| 291 | `blocks.20.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 292 | `blocks.20.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 293 | `blocks.20.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 294 | `blocks.20.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 295 | `blocks.20.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 296 | `blocks.20.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 297 | `blocks.21.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 298 | `blocks.21.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 299 | `blocks.21.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 300 | `blocks.21.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 301 | `blocks.21.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 302 | `blocks.21.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 303 | `blocks.21.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 304 | `blocks.21.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 305 | `blocks.21.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 306 | `blocks.21.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 307 | `blocks.21.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 308 | `blocks.21.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 309 | `blocks.21.modulation` | [9216] | BF16 | 9,216 |
| 310 | `blocks.21.norm2.bias` | [1536] | BF16 | 1,536 |
| 311 | `blocks.21.norm2.weight` | [1536] | BF16 | 1,536 |
| 312 | `blocks.21.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 313 | `blocks.21.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 314 | `blocks.21.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 315 | `blocks.21.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 316 | `blocks.21.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 317 | `blocks.21.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 318 | `blocks.22.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 319 | `blocks.22.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 320 | `blocks.22.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 321 | `blocks.22.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 322 | `blocks.22.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 323 | `blocks.22.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 324 | `blocks.22.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 325 | `blocks.22.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 326 | `blocks.22.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 327 | `blocks.22.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 328 | `blocks.22.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 329 | `blocks.22.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 330 | `blocks.22.modulation` | [9216] | BF16 | 9,216 |
| 331 | `blocks.22.norm2.bias` | [1536] | BF16 | 1,536 |
| 332 | `blocks.22.norm2.weight` | [1536] | BF16 | 1,536 |
| 333 | `blocks.22.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 334 | `blocks.22.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 335 | `blocks.22.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 336 | `blocks.22.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 337 | `blocks.22.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 338 | `blocks.22.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 339 | `blocks.23.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 340 | `blocks.23.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 341 | `blocks.23.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 342 | `blocks.23.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 343 | `blocks.23.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 344 | `blocks.23.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 345 | `blocks.23.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 346 | `blocks.23.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 347 | `blocks.23.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 348 | `blocks.23.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 349 | `blocks.23.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 350 | `blocks.23.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 351 | `blocks.23.modulation` | [9216] | BF16 | 9,216 |
| 352 | `blocks.23.norm2.bias` | [1536] | BF16 | 1,536 |
| 353 | `blocks.23.norm2.weight` | [1536] | BF16 | 1,536 |
| 354 | `blocks.23.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 355 | `blocks.23.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 356 | `blocks.23.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 357 | `blocks.23.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 358 | `blocks.23.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 359 | `blocks.23.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 360 | `blocks.24.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 361 | `blocks.24.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 362 | `blocks.24.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 363 | `blocks.24.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 364 | `blocks.24.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 365 | `blocks.24.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 366 | `blocks.24.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 367 | `blocks.24.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 368 | `blocks.24.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 369 | `blocks.24.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 370 | `blocks.24.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 371 | `blocks.24.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 372 | `blocks.24.modulation` | [9216] | BF16 | 9,216 |
| 373 | `blocks.24.norm2.bias` | [1536] | BF16 | 1,536 |
| 374 | `blocks.24.norm2.weight` | [1536] | BF16 | 1,536 |
| 375 | `blocks.24.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 376 | `blocks.24.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 377 | `blocks.24.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 378 | `blocks.24.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 379 | `blocks.24.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 380 | `blocks.24.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 381 | `blocks.25.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 382 | `blocks.25.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 383 | `blocks.25.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 384 | `blocks.25.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 385 | `blocks.25.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 386 | `blocks.25.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 387 | `blocks.25.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 388 | `blocks.25.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 389 | `blocks.25.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 390 | `blocks.25.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 391 | `blocks.25.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 392 | `blocks.25.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 393 | `blocks.25.modulation` | [9216] | BF16 | 9,216 |
| 394 | `blocks.25.norm2.bias` | [1536] | BF16 | 1,536 |
| 395 | `blocks.25.norm2.weight` | [1536] | BF16 | 1,536 |
| 396 | `blocks.25.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 397 | `blocks.25.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 398 | `blocks.25.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 399 | `blocks.25.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 400 | `blocks.25.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 401 | `blocks.25.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 402 | `blocks.26.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 403 | `blocks.26.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 404 | `blocks.26.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 405 | `blocks.26.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 406 | `blocks.26.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 407 | `blocks.26.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 408 | `blocks.26.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 409 | `blocks.26.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 410 | `blocks.26.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 411 | `blocks.26.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 412 | `blocks.26.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 413 | `blocks.26.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 414 | `blocks.26.modulation` | [9216] | BF16 | 9,216 |
| 415 | `blocks.26.norm2.bias` | [1536] | BF16 | 1,536 |
| 416 | `blocks.26.norm2.weight` | [1536] | BF16 | 1,536 |
| 417 | `blocks.26.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 418 | `blocks.26.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 419 | `blocks.26.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 420 | `blocks.26.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 421 | `blocks.26.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 422 | `blocks.26.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 423 | `blocks.27.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 424 | `blocks.27.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 425 | `blocks.27.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 426 | `blocks.27.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 427 | `blocks.27.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 428 | `blocks.27.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 429 | `blocks.27.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 430 | `blocks.27.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 431 | `blocks.27.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 432 | `blocks.27.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 433 | `blocks.27.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 434 | `blocks.27.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 435 | `blocks.27.modulation` | [9216] | BF16 | 9,216 |
| 436 | `blocks.27.norm2.bias` | [1536] | BF16 | 1,536 |
| 437 | `blocks.27.norm2.weight` | [1536] | BF16 | 1,536 |
| 438 | `blocks.27.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 439 | `blocks.27.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 440 | `blocks.27.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 441 | `blocks.27.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 442 | `blocks.27.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 443 | `blocks.27.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 444 | `blocks.28.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 445 | `blocks.28.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 446 | `blocks.28.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 447 | `blocks.28.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 448 | `blocks.28.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 449 | `blocks.28.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 450 | `blocks.28.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 451 | `blocks.28.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 452 | `blocks.28.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 453 | `blocks.28.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 454 | `blocks.28.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 455 | `blocks.28.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 456 | `blocks.28.modulation` | [9216] | BF16 | 9,216 |
| 457 | `blocks.28.norm2.bias` | [1536] | BF16 | 1,536 |
| 458 | `blocks.28.norm2.weight` | [1536] | BF16 | 1,536 |
| 459 | `blocks.28.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 460 | `blocks.28.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 461 | `blocks.28.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 462 | `blocks.28.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 463 | `blocks.28.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 464 | `blocks.28.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 465 | `blocks.29.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 466 | `blocks.29.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 467 | `blocks.29.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 468 | `blocks.29.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 469 | `blocks.29.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 470 | `blocks.29.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 471 | `blocks.29.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 472 | `blocks.29.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 473 | `blocks.29.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 474 | `blocks.29.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 475 | `blocks.29.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 476 | `blocks.29.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 477 | `blocks.29.modulation` | [9216] | BF16 | 9,216 |
| 478 | `blocks.29.norm2.bias` | [1536] | BF16 | 1,536 |
| 479 | `blocks.29.norm2.weight` | [1536] | BF16 | 1,536 |
| 480 | `blocks.29.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 481 | `blocks.29.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 482 | `blocks.29.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 483 | `blocks.29.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 484 | `blocks.29.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 485 | `blocks.29.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 486 | `blocks.3.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 487 | `blocks.3.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 488 | `blocks.3.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 489 | `blocks.3.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 490 | `blocks.3.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 491 | `blocks.3.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 492 | `blocks.3.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 493 | `blocks.3.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 494 | `blocks.3.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 495 | `blocks.3.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 496 | `blocks.3.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 497 | `blocks.3.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 498 | `blocks.3.modulation` | [9216] | BF16 | 9,216 |
| 499 | `blocks.3.norm2.bias` | [1536] | BF16 | 1,536 |
| 500 | `blocks.3.norm2.weight` | [1536] | BF16 | 1,536 |
| 501 | `blocks.3.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 502 | `blocks.3.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 503 | `blocks.3.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 504 | `blocks.3.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 505 | `blocks.3.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 506 | `blocks.3.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 507 | `blocks.4.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 508 | `blocks.4.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 509 | `blocks.4.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 510 | `blocks.4.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 511 | `blocks.4.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 512 | `blocks.4.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 513 | `blocks.4.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 514 | `blocks.4.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 515 | `blocks.4.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 516 | `blocks.4.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 517 | `blocks.4.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 518 | `blocks.4.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 519 | `blocks.4.modulation` | [9216] | BF16 | 9,216 |
| 520 | `blocks.4.norm2.bias` | [1536] | BF16 | 1,536 |
| 521 | `blocks.4.norm2.weight` | [1536] | BF16 | 1,536 |
| 522 | `blocks.4.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 523 | `blocks.4.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 524 | `blocks.4.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 525 | `blocks.4.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 526 | `blocks.4.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 527 | `blocks.4.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 528 | `blocks.5.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 529 | `blocks.5.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 530 | `blocks.5.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 531 | `blocks.5.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 532 | `blocks.5.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 533 | `blocks.5.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 534 | `blocks.5.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 535 | `blocks.5.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 536 | `blocks.5.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 537 | `blocks.5.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 538 | `blocks.5.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 539 | `blocks.5.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 540 | `blocks.5.modulation` | [9216] | BF16 | 9,216 |
| 541 | `blocks.5.norm2.bias` | [1536] | BF16 | 1,536 |
| 542 | `blocks.5.norm2.weight` | [1536] | BF16 | 1,536 |
| 543 | `blocks.5.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 544 | `blocks.5.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 545 | `blocks.5.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 546 | `blocks.5.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 547 | `blocks.5.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 548 | `blocks.5.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 549 | `blocks.6.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 550 | `blocks.6.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 551 | `blocks.6.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 552 | `blocks.6.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 553 | `blocks.6.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 554 | `blocks.6.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 555 | `blocks.6.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 556 | `blocks.6.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 557 | `blocks.6.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 558 | `blocks.6.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 559 | `blocks.6.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 560 | `blocks.6.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 561 | `blocks.6.modulation` | [9216] | BF16 | 9,216 |
| 562 | `blocks.6.norm2.bias` | [1536] | BF16 | 1,536 |
| 563 | `blocks.6.norm2.weight` | [1536] | BF16 | 1,536 |
| 564 | `blocks.6.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 565 | `blocks.6.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 566 | `blocks.6.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 567 | `blocks.6.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 568 | `blocks.6.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 569 | `blocks.6.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 570 | `blocks.7.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 571 | `blocks.7.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 572 | `blocks.7.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 573 | `blocks.7.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 574 | `blocks.7.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 575 | `blocks.7.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 576 | `blocks.7.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 577 | `blocks.7.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 578 | `blocks.7.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 579 | `blocks.7.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 580 | `blocks.7.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 581 | `blocks.7.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 582 | `blocks.7.modulation` | [9216] | BF16 | 9,216 |
| 583 | `blocks.7.norm2.bias` | [1536] | BF16 | 1,536 |
| 584 | `blocks.7.norm2.weight` | [1536] | BF16 | 1,536 |
| 585 | `blocks.7.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 586 | `blocks.7.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 587 | `blocks.7.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 588 | `blocks.7.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 589 | `blocks.7.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 590 | `blocks.7.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 591 | `blocks.8.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 592 | `blocks.8.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 593 | `blocks.8.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 594 | `blocks.8.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 595 | `blocks.8.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 596 | `blocks.8.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 597 | `blocks.8.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 598 | `blocks.8.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 599 | `blocks.8.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 600 | `blocks.8.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 601 | `blocks.8.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 602 | `blocks.8.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 603 | `blocks.8.modulation` | [9216] | BF16 | 9,216 |
| 604 | `blocks.8.norm2.bias` | [1536] | BF16 | 1,536 |
| 605 | `blocks.8.norm2.weight` | [1536] | BF16 | 1,536 |
| 606 | `blocks.8.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 607 | `blocks.8.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 608 | `blocks.8.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 609 | `blocks.8.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 610 | `blocks.8.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 611 | `blocks.8.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 612 | `blocks.9.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 613 | `blocks.9.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 614 | `blocks.9.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 615 | `blocks.9.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 616 | `blocks.9.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 617 | `blocks.9.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 618 | `blocks.9.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 619 | `blocks.9.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 620 | `blocks.9.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 621 | `blocks.9.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 622 | `blocks.9.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 623 | `blocks.9.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 624 | `blocks.9.modulation` | [9216] | BF16 | 9,216 |
| 625 | `blocks.9.norm2.bias` | [1536] | BF16 | 1,536 |
| 626 | `blocks.9.norm2.weight` | [1536] | BF16 | 1,536 |
| 627 | `blocks.9.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 628 | `blocks.9.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 629 | `blocks.9.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 630 | `blocks.9.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 631 | `blocks.9.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 632 | `blocks.9.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 633 | `input_layer.bias` | [1536] | BF16 | 1,536 |
| 634 | `input_layer.weight` | [1536, 32] | BF16 | 49,152 |
| 635 | `out_layer.bias` | [32] | BF16 | 32 |
| 636 | `out_layer.weight` | [32, 1536] | BF16 | 49,152 |
| 637 | `t_embedder.mlp.0.bias` | [1536] | BF16 | 1,536 |
| 638 | `t_embedder.mlp.0.weight` | [1536, 256] | BF16 | 393,216 |
| 639 | `t_embedder.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 640 | `t_embedder.mlp.2.weight` | [1536, 1536] | BF16 | 2,359,296 |

## `slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors`

**Role:** Stage 2 — Shape SLAT DiT (1024 variant)

Sparse 64³ × 32ch geometry flow model fine-tuned from the 512 variant; used for native 1024 outputs and the HR step in cascades.

Total: **640 parameters**, 1292.25M elements.

Top-level prefixes: `adaLN_modulation` (2), `blocks` (630), `input_layer` (2), `out_layer` (2), `t_embedder` (4)

| # | Parameter | Shape | Dtype | Elements |
|---:|---|---|:---:|---:|
| 1 | `adaLN_modulation.1.bias` | [9216] | BF16 | 9,216 |
| 2 | `adaLN_modulation.1.weight` | [9216, 1536] | BF16 | 14,155,776 |
| 3 | `blocks.0.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 4 | `blocks.0.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 5 | `blocks.0.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 6 | `blocks.0.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 7 | `blocks.0.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 8 | `blocks.0.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 9 | `blocks.0.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 10 | `blocks.0.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 11 | `blocks.0.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 12 | `blocks.0.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 13 | `blocks.0.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 14 | `blocks.0.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 15 | `blocks.0.modulation` | [9216] | BF16 | 9,216 |
| 16 | `blocks.0.norm2.bias` | [1536] | BF16 | 1,536 |
| 17 | `blocks.0.norm2.weight` | [1536] | BF16 | 1,536 |
| 18 | `blocks.0.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 19 | `blocks.0.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 20 | `blocks.0.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 21 | `blocks.0.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 22 | `blocks.0.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 23 | `blocks.0.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 24 | `blocks.1.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 25 | `blocks.1.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 26 | `blocks.1.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 27 | `blocks.1.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 28 | `blocks.1.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 29 | `blocks.1.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 30 | `blocks.1.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 31 | `blocks.1.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 32 | `blocks.1.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 33 | `blocks.1.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 34 | `blocks.1.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 35 | `blocks.1.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 36 | `blocks.1.modulation` | [9216] | BF16 | 9,216 |
| 37 | `blocks.1.norm2.bias` | [1536] | BF16 | 1,536 |
| 38 | `blocks.1.norm2.weight` | [1536] | BF16 | 1,536 |
| 39 | `blocks.1.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 40 | `blocks.1.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 41 | `blocks.1.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 42 | `blocks.1.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 43 | `blocks.1.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 44 | `blocks.1.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 45 | `blocks.10.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 46 | `blocks.10.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 47 | `blocks.10.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 48 | `blocks.10.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 49 | `blocks.10.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 50 | `blocks.10.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 51 | `blocks.10.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 52 | `blocks.10.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 53 | `blocks.10.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 54 | `blocks.10.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 55 | `blocks.10.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 56 | `blocks.10.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 57 | `blocks.10.modulation` | [9216] | BF16 | 9,216 |
| 58 | `blocks.10.norm2.bias` | [1536] | BF16 | 1,536 |
| 59 | `blocks.10.norm2.weight` | [1536] | BF16 | 1,536 |
| 60 | `blocks.10.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 61 | `blocks.10.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 62 | `blocks.10.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 63 | `blocks.10.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 64 | `blocks.10.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 65 | `blocks.10.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 66 | `blocks.11.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 67 | `blocks.11.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 68 | `blocks.11.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 69 | `blocks.11.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 70 | `blocks.11.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 71 | `blocks.11.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 72 | `blocks.11.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 73 | `blocks.11.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 74 | `blocks.11.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 75 | `blocks.11.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 76 | `blocks.11.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 77 | `blocks.11.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 78 | `blocks.11.modulation` | [9216] | BF16 | 9,216 |
| 79 | `blocks.11.norm2.bias` | [1536] | BF16 | 1,536 |
| 80 | `blocks.11.norm2.weight` | [1536] | BF16 | 1,536 |
| 81 | `blocks.11.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 82 | `blocks.11.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 83 | `blocks.11.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 84 | `blocks.11.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 85 | `blocks.11.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 86 | `blocks.11.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 87 | `blocks.12.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 88 | `blocks.12.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 89 | `blocks.12.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 90 | `blocks.12.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 91 | `blocks.12.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 92 | `blocks.12.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 93 | `blocks.12.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 94 | `blocks.12.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 95 | `blocks.12.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 96 | `blocks.12.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 97 | `blocks.12.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 98 | `blocks.12.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 99 | `blocks.12.modulation` | [9216] | BF16 | 9,216 |
| 100 | `blocks.12.norm2.bias` | [1536] | BF16 | 1,536 |
| 101 | `blocks.12.norm2.weight` | [1536] | BF16 | 1,536 |
| 102 | `blocks.12.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 103 | `blocks.12.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 104 | `blocks.12.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 105 | `blocks.12.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 106 | `blocks.12.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 107 | `blocks.12.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 108 | `blocks.13.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 109 | `blocks.13.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 110 | `blocks.13.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 111 | `blocks.13.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 112 | `blocks.13.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 113 | `blocks.13.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 114 | `blocks.13.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 115 | `blocks.13.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 116 | `blocks.13.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 117 | `blocks.13.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 118 | `blocks.13.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 119 | `blocks.13.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 120 | `blocks.13.modulation` | [9216] | BF16 | 9,216 |
| 121 | `blocks.13.norm2.bias` | [1536] | BF16 | 1,536 |
| 122 | `blocks.13.norm2.weight` | [1536] | BF16 | 1,536 |
| 123 | `blocks.13.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 124 | `blocks.13.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 125 | `blocks.13.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 126 | `blocks.13.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 127 | `blocks.13.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 128 | `blocks.13.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 129 | `blocks.14.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 130 | `blocks.14.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 131 | `blocks.14.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 132 | `blocks.14.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 133 | `blocks.14.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 134 | `blocks.14.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 135 | `blocks.14.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 136 | `blocks.14.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 137 | `blocks.14.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 138 | `blocks.14.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 139 | `blocks.14.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 140 | `blocks.14.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 141 | `blocks.14.modulation` | [9216] | BF16 | 9,216 |
| 142 | `blocks.14.norm2.bias` | [1536] | BF16 | 1,536 |
| 143 | `blocks.14.norm2.weight` | [1536] | BF16 | 1,536 |
| 144 | `blocks.14.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 145 | `blocks.14.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 146 | `blocks.14.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 147 | `blocks.14.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 148 | `blocks.14.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 149 | `blocks.14.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 150 | `blocks.15.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 151 | `blocks.15.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 152 | `blocks.15.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 153 | `blocks.15.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 154 | `blocks.15.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 155 | `blocks.15.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 156 | `blocks.15.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 157 | `blocks.15.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 158 | `blocks.15.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 159 | `blocks.15.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 160 | `blocks.15.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 161 | `blocks.15.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 162 | `blocks.15.modulation` | [9216] | BF16 | 9,216 |
| 163 | `blocks.15.norm2.bias` | [1536] | BF16 | 1,536 |
| 164 | `blocks.15.norm2.weight` | [1536] | BF16 | 1,536 |
| 165 | `blocks.15.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 166 | `blocks.15.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 167 | `blocks.15.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 168 | `blocks.15.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 169 | `blocks.15.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 170 | `blocks.15.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 171 | `blocks.16.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 172 | `blocks.16.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 173 | `blocks.16.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 174 | `blocks.16.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 175 | `blocks.16.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 176 | `blocks.16.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 177 | `blocks.16.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 178 | `blocks.16.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 179 | `blocks.16.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 180 | `blocks.16.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 181 | `blocks.16.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 182 | `blocks.16.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 183 | `blocks.16.modulation` | [9216] | BF16 | 9,216 |
| 184 | `blocks.16.norm2.bias` | [1536] | BF16 | 1,536 |
| 185 | `blocks.16.norm2.weight` | [1536] | BF16 | 1,536 |
| 186 | `blocks.16.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 187 | `blocks.16.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 188 | `blocks.16.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 189 | `blocks.16.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 190 | `blocks.16.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 191 | `blocks.16.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 192 | `blocks.17.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 193 | `blocks.17.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 194 | `blocks.17.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 195 | `blocks.17.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 196 | `blocks.17.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 197 | `blocks.17.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 198 | `blocks.17.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 199 | `blocks.17.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 200 | `blocks.17.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 201 | `blocks.17.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 202 | `blocks.17.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 203 | `blocks.17.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 204 | `blocks.17.modulation` | [9216] | BF16 | 9,216 |
| 205 | `blocks.17.norm2.bias` | [1536] | BF16 | 1,536 |
| 206 | `blocks.17.norm2.weight` | [1536] | BF16 | 1,536 |
| 207 | `blocks.17.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 208 | `blocks.17.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 209 | `blocks.17.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 210 | `blocks.17.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 211 | `blocks.17.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 212 | `blocks.17.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 213 | `blocks.18.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 214 | `blocks.18.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 215 | `blocks.18.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 216 | `blocks.18.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 217 | `blocks.18.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 218 | `blocks.18.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 219 | `blocks.18.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 220 | `blocks.18.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 221 | `blocks.18.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 222 | `blocks.18.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 223 | `blocks.18.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 224 | `blocks.18.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 225 | `blocks.18.modulation` | [9216] | BF16 | 9,216 |
| 226 | `blocks.18.norm2.bias` | [1536] | BF16 | 1,536 |
| 227 | `blocks.18.norm2.weight` | [1536] | BF16 | 1,536 |
| 228 | `blocks.18.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 229 | `blocks.18.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 230 | `blocks.18.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 231 | `blocks.18.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 232 | `blocks.18.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 233 | `blocks.18.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 234 | `blocks.19.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 235 | `blocks.19.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 236 | `blocks.19.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 237 | `blocks.19.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 238 | `blocks.19.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 239 | `blocks.19.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 240 | `blocks.19.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 241 | `blocks.19.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 242 | `blocks.19.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 243 | `blocks.19.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 244 | `blocks.19.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 245 | `blocks.19.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 246 | `blocks.19.modulation` | [9216] | BF16 | 9,216 |
| 247 | `blocks.19.norm2.bias` | [1536] | BF16 | 1,536 |
| 248 | `blocks.19.norm2.weight` | [1536] | BF16 | 1,536 |
| 249 | `blocks.19.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 250 | `blocks.19.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 251 | `blocks.19.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 252 | `blocks.19.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 253 | `blocks.19.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 254 | `blocks.19.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 255 | `blocks.2.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 256 | `blocks.2.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 257 | `blocks.2.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 258 | `blocks.2.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 259 | `blocks.2.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 260 | `blocks.2.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 261 | `blocks.2.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 262 | `blocks.2.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 263 | `blocks.2.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 264 | `blocks.2.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 265 | `blocks.2.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 266 | `blocks.2.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 267 | `blocks.2.modulation` | [9216] | BF16 | 9,216 |
| 268 | `blocks.2.norm2.bias` | [1536] | BF16 | 1,536 |
| 269 | `blocks.2.norm2.weight` | [1536] | BF16 | 1,536 |
| 270 | `blocks.2.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 271 | `blocks.2.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 272 | `blocks.2.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 273 | `blocks.2.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 274 | `blocks.2.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 275 | `blocks.2.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 276 | `blocks.20.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 277 | `blocks.20.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 278 | `blocks.20.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 279 | `blocks.20.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 280 | `blocks.20.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 281 | `blocks.20.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 282 | `blocks.20.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 283 | `blocks.20.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 284 | `blocks.20.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 285 | `blocks.20.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 286 | `blocks.20.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 287 | `blocks.20.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 288 | `blocks.20.modulation` | [9216] | BF16 | 9,216 |
| 289 | `blocks.20.norm2.bias` | [1536] | BF16 | 1,536 |
| 290 | `blocks.20.norm2.weight` | [1536] | BF16 | 1,536 |
| 291 | `blocks.20.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 292 | `blocks.20.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 293 | `blocks.20.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 294 | `blocks.20.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 295 | `blocks.20.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 296 | `blocks.20.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 297 | `blocks.21.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 298 | `blocks.21.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 299 | `blocks.21.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 300 | `blocks.21.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 301 | `blocks.21.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 302 | `blocks.21.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 303 | `blocks.21.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 304 | `blocks.21.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 305 | `blocks.21.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 306 | `blocks.21.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 307 | `blocks.21.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 308 | `blocks.21.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 309 | `blocks.21.modulation` | [9216] | BF16 | 9,216 |
| 310 | `blocks.21.norm2.bias` | [1536] | BF16 | 1,536 |
| 311 | `blocks.21.norm2.weight` | [1536] | BF16 | 1,536 |
| 312 | `blocks.21.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 313 | `blocks.21.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 314 | `blocks.21.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 315 | `blocks.21.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 316 | `blocks.21.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 317 | `blocks.21.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 318 | `blocks.22.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 319 | `blocks.22.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 320 | `blocks.22.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 321 | `blocks.22.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 322 | `blocks.22.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 323 | `blocks.22.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 324 | `blocks.22.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 325 | `blocks.22.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 326 | `blocks.22.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 327 | `blocks.22.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 328 | `blocks.22.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 329 | `blocks.22.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 330 | `blocks.22.modulation` | [9216] | BF16 | 9,216 |
| 331 | `blocks.22.norm2.bias` | [1536] | BF16 | 1,536 |
| 332 | `blocks.22.norm2.weight` | [1536] | BF16 | 1,536 |
| 333 | `blocks.22.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 334 | `blocks.22.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 335 | `blocks.22.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 336 | `blocks.22.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 337 | `blocks.22.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 338 | `blocks.22.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 339 | `blocks.23.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 340 | `blocks.23.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 341 | `blocks.23.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 342 | `blocks.23.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 343 | `blocks.23.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 344 | `blocks.23.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 345 | `blocks.23.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 346 | `blocks.23.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 347 | `blocks.23.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 348 | `blocks.23.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 349 | `blocks.23.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 350 | `blocks.23.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 351 | `blocks.23.modulation` | [9216] | BF16 | 9,216 |
| 352 | `blocks.23.norm2.bias` | [1536] | BF16 | 1,536 |
| 353 | `blocks.23.norm2.weight` | [1536] | BF16 | 1,536 |
| 354 | `blocks.23.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 355 | `blocks.23.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 356 | `blocks.23.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 357 | `blocks.23.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 358 | `blocks.23.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 359 | `blocks.23.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 360 | `blocks.24.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 361 | `blocks.24.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 362 | `blocks.24.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 363 | `blocks.24.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 364 | `blocks.24.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 365 | `blocks.24.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 366 | `blocks.24.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 367 | `blocks.24.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 368 | `blocks.24.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 369 | `blocks.24.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 370 | `blocks.24.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 371 | `blocks.24.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 372 | `blocks.24.modulation` | [9216] | BF16 | 9,216 |
| 373 | `blocks.24.norm2.bias` | [1536] | BF16 | 1,536 |
| 374 | `blocks.24.norm2.weight` | [1536] | BF16 | 1,536 |
| 375 | `blocks.24.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 376 | `blocks.24.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 377 | `blocks.24.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 378 | `blocks.24.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 379 | `blocks.24.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 380 | `blocks.24.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 381 | `blocks.25.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 382 | `blocks.25.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 383 | `blocks.25.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 384 | `blocks.25.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 385 | `blocks.25.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 386 | `blocks.25.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 387 | `blocks.25.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 388 | `blocks.25.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 389 | `blocks.25.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 390 | `blocks.25.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 391 | `blocks.25.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 392 | `blocks.25.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 393 | `blocks.25.modulation` | [9216] | BF16 | 9,216 |
| 394 | `blocks.25.norm2.bias` | [1536] | BF16 | 1,536 |
| 395 | `blocks.25.norm2.weight` | [1536] | BF16 | 1,536 |
| 396 | `blocks.25.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 397 | `blocks.25.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 398 | `blocks.25.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 399 | `blocks.25.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 400 | `blocks.25.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 401 | `blocks.25.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 402 | `blocks.26.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 403 | `blocks.26.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 404 | `blocks.26.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 405 | `blocks.26.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 406 | `blocks.26.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 407 | `blocks.26.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 408 | `blocks.26.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 409 | `blocks.26.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 410 | `blocks.26.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 411 | `blocks.26.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 412 | `blocks.26.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 413 | `blocks.26.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 414 | `blocks.26.modulation` | [9216] | BF16 | 9,216 |
| 415 | `blocks.26.norm2.bias` | [1536] | BF16 | 1,536 |
| 416 | `blocks.26.norm2.weight` | [1536] | BF16 | 1,536 |
| 417 | `blocks.26.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 418 | `blocks.26.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 419 | `blocks.26.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 420 | `blocks.26.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 421 | `blocks.26.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 422 | `blocks.26.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 423 | `blocks.27.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 424 | `blocks.27.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 425 | `blocks.27.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 426 | `blocks.27.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 427 | `blocks.27.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 428 | `blocks.27.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 429 | `blocks.27.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 430 | `blocks.27.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 431 | `blocks.27.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 432 | `blocks.27.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 433 | `blocks.27.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 434 | `blocks.27.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 435 | `blocks.27.modulation` | [9216] | BF16 | 9,216 |
| 436 | `blocks.27.norm2.bias` | [1536] | BF16 | 1,536 |
| 437 | `blocks.27.norm2.weight` | [1536] | BF16 | 1,536 |
| 438 | `blocks.27.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 439 | `blocks.27.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 440 | `blocks.27.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 441 | `blocks.27.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 442 | `blocks.27.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 443 | `blocks.27.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 444 | `blocks.28.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 445 | `blocks.28.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 446 | `blocks.28.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 447 | `blocks.28.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 448 | `blocks.28.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 449 | `blocks.28.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 450 | `blocks.28.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 451 | `blocks.28.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 452 | `blocks.28.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 453 | `blocks.28.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 454 | `blocks.28.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 455 | `blocks.28.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 456 | `blocks.28.modulation` | [9216] | BF16 | 9,216 |
| 457 | `blocks.28.norm2.bias` | [1536] | BF16 | 1,536 |
| 458 | `blocks.28.norm2.weight` | [1536] | BF16 | 1,536 |
| 459 | `blocks.28.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 460 | `blocks.28.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 461 | `blocks.28.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 462 | `blocks.28.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 463 | `blocks.28.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 464 | `blocks.28.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 465 | `blocks.29.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 466 | `blocks.29.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 467 | `blocks.29.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 468 | `blocks.29.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 469 | `blocks.29.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 470 | `blocks.29.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 471 | `blocks.29.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 472 | `blocks.29.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 473 | `blocks.29.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 474 | `blocks.29.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 475 | `blocks.29.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 476 | `blocks.29.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 477 | `blocks.29.modulation` | [9216] | BF16 | 9,216 |
| 478 | `blocks.29.norm2.bias` | [1536] | BF16 | 1,536 |
| 479 | `blocks.29.norm2.weight` | [1536] | BF16 | 1,536 |
| 480 | `blocks.29.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 481 | `blocks.29.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 482 | `blocks.29.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 483 | `blocks.29.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 484 | `blocks.29.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 485 | `blocks.29.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 486 | `blocks.3.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 487 | `blocks.3.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 488 | `blocks.3.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 489 | `blocks.3.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 490 | `blocks.3.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 491 | `blocks.3.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 492 | `blocks.3.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 493 | `blocks.3.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 494 | `blocks.3.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 495 | `blocks.3.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 496 | `blocks.3.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 497 | `blocks.3.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 498 | `blocks.3.modulation` | [9216] | BF16 | 9,216 |
| 499 | `blocks.3.norm2.bias` | [1536] | BF16 | 1,536 |
| 500 | `blocks.3.norm2.weight` | [1536] | BF16 | 1,536 |
| 501 | `blocks.3.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 502 | `blocks.3.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 503 | `blocks.3.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 504 | `blocks.3.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 505 | `blocks.3.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 506 | `blocks.3.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 507 | `blocks.4.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 508 | `blocks.4.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 509 | `blocks.4.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 510 | `blocks.4.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 511 | `blocks.4.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 512 | `blocks.4.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 513 | `blocks.4.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 514 | `blocks.4.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 515 | `blocks.4.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 516 | `blocks.4.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 517 | `blocks.4.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 518 | `blocks.4.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 519 | `blocks.4.modulation` | [9216] | BF16 | 9,216 |
| 520 | `blocks.4.norm2.bias` | [1536] | BF16 | 1,536 |
| 521 | `blocks.4.norm2.weight` | [1536] | BF16 | 1,536 |
| 522 | `blocks.4.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 523 | `blocks.4.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 524 | `blocks.4.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 525 | `blocks.4.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 526 | `blocks.4.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 527 | `blocks.4.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 528 | `blocks.5.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 529 | `blocks.5.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 530 | `blocks.5.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 531 | `blocks.5.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 532 | `blocks.5.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 533 | `blocks.5.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 534 | `blocks.5.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 535 | `blocks.5.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 536 | `blocks.5.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 537 | `blocks.5.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 538 | `blocks.5.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 539 | `blocks.5.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 540 | `blocks.5.modulation` | [9216] | BF16 | 9,216 |
| 541 | `blocks.5.norm2.bias` | [1536] | BF16 | 1,536 |
| 542 | `blocks.5.norm2.weight` | [1536] | BF16 | 1,536 |
| 543 | `blocks.5.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 544 | `blocks.5.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 545 | `blocks.5.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 546 | `blocks.5.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 547 | `blocks.5.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 548 | `blocks.5.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 549 | `blocks.6.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 550 | `blocks.6.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 551 | `blocks.6.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 552 | `blocks.6.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 553 | `blocks.6.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 554 | `blocks.6.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 555 | `blocks.6.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 556 | `blocks.6.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 557 | `blocks.6.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 558 | `blocks.6.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 559 | `blocks.6.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 560 | `blocks.6.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 561 | `blocks.6.modulation` | [9216] | BF16 | 9,216 |
| 562 | `blocks.6.norm2.bias` | [1536] | BF16 | 1,536 |
| 563 | `blocks.6.norm2.weight` | [1536] | BF16 | 1,536 |
| 564 | `blocks.6.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 565 | `blocks.6.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 566 | `blocks.6.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 567 | `blocks.6.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 568 | `blocks.6.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 569 | `blocks.6.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 570 | `blocks.7.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 571 | `blocks.7.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 572 | `blocks.7.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 573 | `blocks.7.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 574 | `blocks.7.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 575 | `blocks.7.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 576 | `blocks.7.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 577 | `blocks.7.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 578 | `blocks.7.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 579 | `blocks.7.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 580 | `blocks.7.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 581 | `blocks.7.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 582 | `blocks.7.modulation` | [9216] | BF16 | 9,216 |
| 583 | `blocks.7.norm2.bias` | [1536] | BF16 | 1,536 |
| 584 | `blocks.7.norm2.weight` | [1536] | BF16 | 1,536 |
| 585 | `blocks.7.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 586 | `blocks.7.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 587 | `blocks.7.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 588 | `blocks.7.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 589 | `blocks.7.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 590 | `blocks.7.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 591 | `blocks.8.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 592 | `blocks.8.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 593 | `blocks.8.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 594 | `blocks.8.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 595 | `blocks.8.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 596 | `blocks.8.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 597 | `blocks.8.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 598 | `blocks.8.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 599 | `blocks.8.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 600 | `blocks.8.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 601 | `blocks.8.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 602 | `blocks.8.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 603 | `blocks.8.modulation` | [9216] | BF16 | 9,216 |
| 604 | `blocks.8.norm2.bias` | [1536] | BF16 | 1,536 |
| 605 | `blocks.8.norm2.weight` | [1536] | BF16 | 1,536 |
| 606 | `blocks.8.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 607 | `blocks.8.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 608 | `blocks.8.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 609 | `blocks.8.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 610 | `blocks.8.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 611 | `blocks.8.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 612 | `blocks.9.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 613 | `blocks.9.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 614 | `blocks.9.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 615 | `blocks.9.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 616 | `blocks.9.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 617 | `blocks.9.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 618 | `blocks.9.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 619 | `blocks.9.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 620 | `blocks.9.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 621 | `blocks.9.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 622 | `blocks.9.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 623 | `blocks.9.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 624 | `blocks.9.modulation` | [9216] | BF16 | 9,216 |
| 625 | `blocks.9.norm2.bias` | [1536] | BF16 | 1,536 |
| 626 | `blocks.9.norm2.weight` | [1536] | BF16 | 1,536 |
| 627 | `blocks.9.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 628 | `blocks.9.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 629 | `blocks.9.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 630 | `blocks.9.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 631 | `blocks.9.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 632 | `blocks.9.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 633 | `input_layer.bias` | [1536] | BF16 | 1,536 |
| 634 | `input_layer.weight` | [1536, 32] | BF16 | 49,152 |
| 635 | `out_layer.bias` | [32] | BF16 | 32 |
| 636 | `out_layer.weight` | [32, 1536] | BF16 | 49,152 |
| 637 | `t_embedder.mlp.0.bias` | [1536] | BF16 | 1,536 |
| 638 | `t_embedder.mlp.0.weight` | [1536, 256] | BF16 | 393,216 |
| 639 | `t_embedder.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 640 | `t_embedder.mlp.2.weight` | [1536, 1536] | BF16 | 2,359,296 |

## `slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors`

**Role:** Stage 3 — Texture SLAT DiT (512 variant)

Sparse 32³ × 64ch material flow model; in_channels=64 because the shape latent is concatenated channel-wise as conditioning.

Total: **640 parameters**, 1292.30M elements.

Top-level prefixes: `adaLN_modulation` (2), `blocks` (630), `input_layer` (2), `out_layer` (2), `t_embedder` (4)

| # | Parameter | Shape | Dtype | Elements |
|---:|---|---|:---:|---:|
| 1 | `adaLN_modulation.1.bias` | [9216] | BF16 | 9,216 |
| 2 | `adaLN_modulation.1.weight` | [9216, 1536] | BF16 | 14,155,776 |
| 3 | `blocks.0.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 4 | `blocks.0.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 5 | `blocks.0.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 6 | `blocks.0.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 7 | `blocks.0.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 8 | `blocks.0.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 9 | `blocks.0.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 10 | `blocks.0.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 11 | `blocks.0.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 12 | `blocks.0.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 13 | `blocks.0.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 14 | `blocks.0.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 15 | `blocks.0.modulation` | [9216] | BF16 | 9,216 |
| 16 | `blocks.0.norm2.bias` | [1536] | BF16 | 1,536 |
| 17 | `blocks.0.norm2.weight` | [1536] | BF16 | 1,536 |
| 18 | `blocks.0.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 19 | `blocks.0.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 20 | `blocks.0.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 21 | `blocks.0.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 22 | `blocks.0.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 23 | `blocks.0.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 24 | `blocks.1.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 25 | `blocks.1.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 26 | `blocks.1.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 27 | `blocks.1.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 28 | `blocks.1.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 29 | `blocks.1.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 30 | `blocks.1.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 31 | `blocks.1.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 32 | `blocks.1.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 33 | `blocks.1.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 34 | `blocks.1.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 35 | `blocks.1.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 36 | `blocks.1.modulation` | [9216] | BF16 | 9,216 |
| 37 | `blocks.1.norm2.bias` | [1536] | BF16 | 1,536 |
| 38 | `blocks.1.norm2.weight` | [1536] | BF16 | 1,536 |
| 39 | `blocks.1.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 40 | `blocks.1.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 41 | `blocks.1.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 42 | `blocks.1.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 43 | `blocks.1.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 44 | `blocks.1.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 45 | `blocks.10.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 46 | `blocks.10.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 47 | `blocks.10.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 48 | `blocks.10.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 49 | `blocks.10.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 50 | `blocks.10.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 51 | `blocks.10.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 52 | `blocks.10.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 53 | `blocks.10.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 54 | `blocks.10.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 55 | `blocks.10.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 56 | `blocks.10.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 57 | `blocks.10.modulation` | [9216] | BF16 | 9,216 |
| 58 | `blocks.10.norm2.bias` | [1536] | BF16 | 1,536 |
| 59 | `blocks.10.norm2.weight` | [1536] | BF16 | 1,536 |
| 60 | `blocks.10.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 61 | `blocks.10.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 62 | `blocks.10.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 63 | `blocks.10.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 64 | `blocks.10.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 65 | `blocks.10.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 66 | `blocks.11.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 67 | `blocks.11.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 68 | `blocks.11.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 69 | `blocks.11.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 70 | `blocks.11.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 71 | `blocks.11.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 72 | `blocks.11.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 73 | `blocks.11.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 74 | `blocks.11.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 75 | `blocks.11.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 76 | `blocks.11.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 77 | `blocks.11.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 78 | `blocks.11.modulation` | [9216] | BF16 | 9,216 |
| 79 | `blocks.11.norm2.bias` | [1536] | BF16 | 1,536 |
| 80 | `blocks.11.norm2.weight` | [1536] | BF16 | 1,536 |
| 81 | `blocks.11.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 82 | `blocks.11.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 83 | `blocks.11.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 84 | `blocks.11.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 85 | `blocks.11.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 86 | `blocks.11.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 87 | `blocks.12.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 88 | `blocks.12.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 89 | `blocks.12.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 90 | `blocks.12.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 91 | `blocks.12.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 92 | `blocks.12.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 93 | `blocks.12.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 94 | `blocks.12.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 95 | `blocks.12.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 96 | `blocks.12.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 97 | `blocks.12.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 98 | `blocks.12.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 99 | `blocks.12.modulation` | [9216] | BF16 | 9,216 |
| 100 | `blocks.12.norm2.bias` | [1536] | BF16 | 1,536 |
| 101 | `blocks.12.norm2.weight` | [1536] | BF16 | 1,536 |
| 102 | `blocks.12.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 103 | `blocks.12.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 104 | `blocks.12.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 105 | `blocks.12.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 106 | `blocks.12.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 107 | `blocks.12.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 108 | `blocks.13.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 109 | `blocks.13.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 110 | `blocks.13.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 111 | `blocks.13.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 112 | `blocks.13.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 113 | `blocks.13.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 114 | `blocks.13.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 115 | `blocks.13.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 116 | `blocks.13.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 117 | `blocks.13.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 118 | `blocks.13.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 119 | `blocks.13.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 120 | `blocks.13.modulation` | [9216] | BF16 | 9,216 |
| 121 | `blocks.13.norm2.bias` | [1536] | BF16 | 1,536 |
| 122 | `blocks.13.norm2.weight` | [1536] | BF16 | 1,536 |
| 123 | `blocks.13.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 124 | `blocks.13.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 125 | `blocks.13.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 126 | `blocks.13.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 127 | `blocks.13.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 128 | `blocks.13.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 129 | `blocks.14.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 130 | `blocks.14.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 131 | `blocks.14.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 132 | `blocks.14.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 133 | `blocks.14.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 134 | `blocks.14.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 135 | `blocks.14.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 136 | `blocks.14.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 137 | `blocks.14.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 138 | `blocks.14.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 139 | `blocks.14.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 140 | `blocks.14.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 141 | `blocks.14.modulation` | [9216] | BF16 | 9,216 |
| 142 | `blocks.14.norm2.bias` | [1536] | BF16 | 1,536 |
| 143 | `blocks.14.norm2.weight` | [1536] | BF16 | 1,536 |
| 144 | `blocks.14.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 145 | `blocks.14.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 146 | `blocks.14.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 147 | `blocks.14.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 148 | `blocks.14.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 149 | `blocks.14.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 150 | `blocks.15.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 151 | `blocks.15.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 152 | `blocks.15.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 153 | `blocks.15.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 154 | `blocks.15.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 155 | `blocks.15.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 156 | `blocks.15.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 157 | `blocks.15.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 158 | `blocks.15.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 159 | `blocks.15.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 160 | `blocks.15.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 161 | `blocks.15.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 162 | `blocks.15.modulation` | [9216] | BF16 | 9,216 |
| 163 | `blocks.15.norm2.bias` | [1536] | BF16 | 1,536 |
| 164 | `blocks.15.norm2.weight` | [1536] | BF16 | 1,536 |
| 165 | `blocks.15.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 166 | `blocks.15.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 167 | `blocks.15.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 168 | `blocks.15.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 169 | `blocks.15.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 170 | `blocks.15.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 171 | `blocks.16.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 172 | `blocks.16.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 173 | `blocks.16.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 174 | `blocks.16.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 175 | `blocks.16.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 176 | `blocks.16.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 177 | `blocks.16.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 178 | `blocks.16.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 179 | `blocks.16.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 180 | `blocks.16.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 181 | `blocks.16.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 182 | `blocks.16.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 183 | `blocks.16.modulation` | [9216] | BF16 | 9,216 |
| 184 | `blocks.16.norm2.bias` | [1536] | BF16 | 1,536 |
| 185 | `blocks.16.norm2.weight` | [1536] | BF16 | 1,536 |
| 186 | `blocks.16.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 187 | `blocks.16.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 188 | `blocks.16.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 189 | `blocks.16.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 190 | `blocks.16.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 191 | `blocks.16.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 192 | `blocks.17.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 193 | `blocks.17.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 194 | `blocks.17.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 195 | `blocks.17.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 196 | `blocks.17.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 197 | `blocks.17.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 198 | `blocks.17.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 199 | `blocks.17.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 200 | `blocks.17.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 201 | `blocks.17.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 202 | `blocks.17.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 203 | `blocks.17.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 204 | `blocks.17.modulation` | [9216] | BF16 | 9,216 |
| 205 | `blocks.17.norm2.bias` | [1536] | BF16 | 1,536 |
| 206 | `blocks.17.norm2.weight` | [1536] | BF16 | 1,536 |
| 207 | `blocks.17.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 208 | `blocks.17.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 209 | `blocks.17.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 210 | `blocks.17.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 211 | `blocks.17.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 212 | `blocks.17.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 213 | `blocks.18.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 214 | `blocks.18.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 215 | `blocks.18.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 216 | `blocks.18.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 217 | `blocks.18.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 218 | `blocks.18.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 219 | `blocks.18.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 220 | `blocks.18.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 221 | `blocks.18.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 222 | `blocks.18.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 223 | `blocks.18.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 224 | `blocks.18.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 225 | `blocks.18.modulation` | [9216] | BF16 | 9,216 |
| 226 | `blocks.18.norm2.bias` | [1536] | BF16 | 1,536 |
| 227 | `blocks.18.norm2.weight` | [1536] | BF16 | 1,536 |
| 228 | `blocks.18.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 229 | `blocks.18.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 230 | `blocks.18.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 231 | `blocks.18.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 232 | `blocks.18.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 233 | `blocks.18.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 234 | `blocks.19.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 235 | `blocks.19.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 236 | `blocks.19.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 237 | `blocks.19.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 238 | `blocks.19.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 239 | `blocks.19.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 240 | `blocks.19.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 241 | `blocks.19.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 242 | `blocks.19.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 243 | `blocks.19.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 244 | `blocks.19.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 245 | `blocks.19.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 246 | `blocks.19.modulation` | [9216] | BF16 | 9,216 |
| 247 | `blocks.19.norm2.bias` | [1536] | BF16 | 1,536 |
| 248 | `blocks.19.norm2.weight` | [1536] | BF16 | 1,536 |
| 249 | `blocks.19.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 250 | `blocks.19.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 251 | `blocks.19.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 252 | `blocks.19.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 253 | `blocks.19.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 254 | `blocks.19.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 255 | `blocks.2.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 256 | `blocks.2.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 257 | `blocks.2.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 258 | `blocks.2.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 259 | `blocks.2.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 260 | `blocks.2.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 261 | `blocks.2.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 262 | `blocks.2.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 263 | `blocks.2.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 264 | `blocks.2.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 265 | `blocks.2.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 266 | `blocks.2.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 267 | `blocks.2.modulation` | [9216] | BF16 | 9,216 |
| 268 | `blocks.2.norm2.bias` | [1536] | BF16 | 1,536 |
| 269 | `blocks.2.norm2.weight` | [1536] | BF16 | 1,536 |
| 270 | `blocks.2.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 271 | `blocks.2.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 272 | `blocks.2.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 273 | `blocks.2.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 274 | `blocks.2.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 275 | `blocks.2.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 276 | `blocks.20.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 277 | `blocks.20.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 278 | `blocks.20.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 279 | `blocks.20.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 280 | `blocks.20.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 281 | `blocks.20.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 282 | `blocks.20.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 283 | `blocks.20.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 284 | `blocks.20.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 285 | `blocks.20.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 286 | `blocks.20.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 287 | `blocks.20.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 288 | `blocks.20.modulation` | [9216] | BF16 | 9,216 |
| 289 | `blocks.20.norm2.bias` | [1536] | BF16 | 1,536 |
| 290 | `blocks.20.norm2.weight` | [1536] | BF16 | 1,536 |
| 291 | `blocks.20.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 292 | `blocks.20.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 293 | `blocks.20.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 294 | `blocks.20.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 295 | `blocks.20.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 296 | `blocks.20.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 297 | `blocks.21.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 298 | `blocks.21.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 299 | `blocks.21.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 300 | `blocks.21.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 301 | `blocks.21.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 302 | `blocks.21.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 303 | `blocks.21.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 304 | `blocks.21.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 305 | `blocks.21.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 306 | `blocks.21.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 307 | `blocks.21.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 308 | `blocks.21.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 309 | `blocks.21.modulation` | [9216] | BF16 | 9,216 |
| 310 | `blocks.21.norm2.bias` | [1536] | BF16 | 1,536 |
| 311 | `blocks.21.norm2.weight` | [1536] | BF16 | 1,536 |
| 312 | `blocks.21.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 313 | `blocks.21.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 314 | `blocks.21.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 315 | `blocks.21.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 316 | `blocks.21.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 317 | `blocks.21.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 318 | `blocks.22.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 319 | `blocks.22.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 320 | `blocks.22.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 321 | `blocks.22.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 322 | `blocks.22.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 323 | `blocks.22.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 324 | `blocks.22.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 325 | `blocks.22.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 326 | `blocks.22.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 327 | `blocks.22.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 328 | `blocks.22.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 329 | `blocks.22.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 330 | `blocks.22.modulation` | [9216] | BF16 | 9,216 |
| 331 | `blocks.22.norm2.bias` | [1536] | BF16 | 1,536 |
| 332 | `blocks.22.norm2.weight` | [1536] | BF16 | 1,536 |
| 333 | `blocks.22.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 334 | `blocks.22.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 335 | `blocks.22.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 336 | `blocks.22.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 337 | `blocks.22.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 338 | `blocks.22.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 339 | `blocks.23.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 340 | `blocks.23.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 341 | `blocks.23.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 342 | `blocks.23.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 343 | `blocks.23.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 344 | `blocks.23.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 345 | `blocks.23.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 346 | `blocks.23.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 347 | `blocks.23.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 348 | `blocks.23.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 349 | `blocks.23.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 350 | `blocks.23.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 351 | `blocks.23.modulation` | [9216] | BF16 | 9,216 |
| 352 | `blocks.23.norm2.bias` | [1536] | BF16 | 1,536 |
| 353 | `blocks.23.norm2.weight` | [1536] | BF16 | 1,536 |
| 354 | `blocks.23.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 355 | `blocks.23.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 356 | `blocks.23.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 357 | `blocks.23.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 358 | `blocks.23.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 359 | `blocks.23.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 360 | `blocks.24.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 361 | `blocks.24.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 362 | `blocks.24.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 363 | `blocks.24.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 364 | `blocks.24.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 365 | `blocks.24.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 366 | `blocks.24.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 367 | `blocks.24.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 368 | `blocks.24.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 369 | `blocks.24.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 370 | `blocks.24.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 371 | `blocks.24.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 372 | `blocks.24.modulation` | [9216] | BF16 | 9,216 |
| 373 | `blocks.24.norm2.bias` | [1536] | BF16 | 1,536 |
| 374 | `blocks.24.norm2.weight` | [1536] | BF16 | 1,536 |
| 375 | `blocks.24.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 376 | `blocks.24.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 377 | `blocks.24.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 378 | `blocks.24.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 379 | `blocks.24.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 380 | `blocks.24.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 381 | `blocks.25.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 382 | `blocks.25.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 383 | `blocks.25.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 384 | `blocks.25.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 385 | `blocks.25.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 386 | `blocks.25.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 387 | `blocks.25.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 388 | `blocks.25.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 389 | `blocks.25.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 390 | `blocks.25.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 391 | `blocks.25.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 392 | `blocks.25.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 393 | `blocks.25.modulation` | [9216] | BF16 | 9,216 |
| 394 | `blocks.25.norm2.bias` | [1536] | BF16 | 1,536 |
| 395 | `blocks.25.norm2.weight` | [1536] | BF16 | 1,536 |
| 396 | `blocks.25.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 397 | `blocks.25.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 398 | `blocks.25.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 399 | `blocks.25.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 400 | `blocks.25.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 401 | `blocks.25.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 402 | `blocks.26.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 403 | `blocks.26.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 404 | `blocks.26.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 405 | `blocks.26.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 406 | `blocks.26.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 407 | `blocks.26.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 408 | `blocks.26.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 409 | `blocks.26.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 410 | `blocks.26.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 411 | `blocks.26.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 412 | `blocks.26.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 413 | `blocks.26.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 414 | `blocks.26.modulation` | [9216] | BF16 | 9,216 |
| 415 | `blocks.26.norm2.bias` | [1536] | BF16 | 1,536 |
| 416 | `blocks.26.norm2.weight` | [1536] | BF16 | 1,536 |
| 417 | `blocks.26.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 418 | `blocks.26.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 419 | `blocks.26.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 420 | `blocks.26.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 421 | `blocks.26.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 422 | `blocks.26.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 423 | `blocks.27.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 424 | `blocks.27.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 425 | `blocks.27.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 426 | `blocks.27.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 427 | `blocks.27.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 428 | `blocks.27.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 429 | `blocks.27.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 430 | `blocks.27.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 431 | `blocks.27.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 432 | `blocks.27.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 433 | `blocks.27.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 434 | `blocks.27.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 435 | `blocks.27.modulation` | [9216] | BF16 | 9,216 |
| 436 | `blocks.27.norm2.bias` | [1536] | BF16 | 1,536 |
| 437 | `blocks.27.norm2.weight` | [1536] | BF16 | 1,536 |
| 438 | `blocks.27.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 439 | `blocks.27.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 440 | `blocks.27.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 441 | `blocks.27.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 442 | `blocks.27.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 443 | `blocks.27.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 444 | `blocks.28.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 445 | `blocks.28.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 446 | `blocks.28.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 447 | `blocks.28.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 448 | `blocks.28.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 449 | `blocks.28.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 450 | `blocks.28.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 451 | `blocks.28.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 452 | `blocks.28.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 453 | `blocks.28.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 454 | `blocks.28.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 455 | `blocks.28.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 456 | `blocks.28.modulation` | [9216] | BF16 | 9,216 |
| 457 | `blocks.28.norm2.bias` | [1536] | BF16 | 1,536 |
| 458 | `blocks.28.norm2.weight` | [1536] | BF16 | 1,536 |
| 459 | `blocks.28.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 460 | `blocks.28.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 461 | `blocks.28.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 462 | `blocks.28.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 463 | `blocks.28.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 464 | `blocks.28.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 465 | `blocks.29.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 466 | `blocks.29.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 467 | `blocks.29.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 468 | `blocks.29.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 469 | `blocks.29.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 470 | `blocks.29.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 471 | `blocks.29.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 472 | `blocks.29.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 473 | `blocks.29.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 474 | `blocks.29.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 475 | `blocks.29.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 476 | `blocks.29.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 477 | `blocks.29.modulation` | [9216] | BF16 | 9,216 |
| 478 | `blocks.29.norm2.bias` | [1536] | BF16 | 1,536 |
| 479 | `blocks.29.norm2.weight` | [1536] | BF16 | 1,536 |
| 480 | `blocks.29.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 481 | `blocks.29.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 482 | `blocks.29.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 483 | `blocks.29.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 484 | `blocks.29.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 485 | `blocks.29.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 486 | `blocks.3.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 487 | `blocks.3.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 488 | `blocks.3.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 489 | `blocks.3.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 490 | `blocks.3.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 491 | `blocks.3.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 492 | `blocks.3.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 493 | `blocks.3.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 494 | `blocks.3.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 495 | `blocks.3.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 496 | `blocks.3.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 497 | `blocks.3.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 498 | `blocks.3.modulation` | [9216] | BF16 | 9,216 |
| 499 | `blocks.3.norm2.bias` | [1536] | BF16 | 1,536 |
| 500 | `blocks.3.norm2.weight` | [1536] | BF16 | 1,536 |
| 501 | `blocks.3.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 502 | `blocks.3.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 503 | `blocks.3.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 504 | `blocks.3.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 505 | `blocks.3.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 506 | `blocks.3.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 507 | `blocks.4.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 508 | `blocks.4.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 509 | `blocks.4.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 510 | `blocks.4.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 511 | `blocks.4.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 512 | `blocks.4.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 513 | `blocks.4.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 514 | `blocks.4.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 515 | `blocks.4.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 516 | `blocks.4.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 517 | `blocks.4.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 518 | `blocks.4.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 519 | `blocks.4.modulation` | [9216] | BF16 | 9,216 |
| 520 | `blocks.4.norm2.bias` | [1536] | BF16 | 1,536 |
| 521 | `blocks.4.norm2.weight` | [1536] | BF16 | 1,536 |
| 522 | `blocks.4.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 523 | `blocks.4.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 524 | `blocks.4.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 525 | `blocks.4.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 526 | `blocks.4.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 527 | `blocks.4.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 528 | `blocks.5.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 529 | `blocks.5.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 530 | `blocks.5.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 531 | `blocks.5.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 532 | `blocks.5.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 533 | `blocks.5.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 534 | `blocks.5.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 535 | `blocks.5.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 536 | `blocks.5.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 537 | `blocks.5.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 538 | `blocks.5.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 539 | `blocks.5.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 540 | `blocks.5.modulation` | [9216] | BF16 | 9,216 |
| 541 | `blocks.5.norm2.bias` | [1536] | BF16 | 1,536 |
| 542 | `blocks.5.norm2.weight` | [1536] | BF16 | 1,536 |
| 543 | `blocks.5.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 544 | `blocks.5.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 545 | `blocks.5.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 546 | `blocks.5.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 547 | `blocks.5.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 548 | `blocks.5.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 549 | `blocks.6.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 550 | `blocks.6.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 551 | `blocks.6.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 552 | `blocks.6.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 553 | `blocks.6.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 554 | `blocks.6.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 555 | `blocks.6.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 556 | `blocks.6.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 557 | `blocks.6.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 558 | `blocks.6.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 559 | `blocks.6.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 560 | `blocks.6.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 561 | `blocks.6.modulation` | [9216] | BF16 | 9,216 |
| 562 | `blocks.6.norm2.bias` | [1536] | BF16 | 1,536 |
| 563 | `blocks.6.norm2.weight` | [1536] | BF16 | 1,536 |
| 564 | `blocks.6.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 565 | `blocks.6.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 566 | `blocks.6.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 567 | `blocks.6.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 568 | `blocks.6.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 569 | `blocks.6.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 570 | `blocks.7.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 571 | `blocks.7.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 572 | `blocks.7.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 573 | `blocks.7.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 574 | `blocks.7.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 575 | `blocks.7.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 576 | `blocks.7.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 577 | `blocks.7.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 578 | `blocks.7.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 579 | `blocks.7.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 580 | `blocks.7.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 581 | `blocks.7.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 582 | `blocks.7.modulation` | [9216] | BF16 | 9,216 |
| 583 | `blocks.7.norm2.bias` | [1536] | BF16 | 1,536 |
| 584 | `blocks.7.norm2.weight` | [1536] | BF16 | 1,536 |
| 585 | `blocks.7.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 586 | `blocks.7.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 587 | `blocks.7.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 588 | `blocks.7.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 589 | `blocks.7.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 590 | `blocks.7.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 591 | `blocks.8.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 592 | `blocks.8.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 593 | `blocks.8.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 594 | `blocks.8.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 595 | `blocks.8.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 596 | `blocks.8.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 597 | `blocks.8.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 598 | `blocks.8.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 599 | `blocks.8.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 600 | `blocks.8.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 601 | `blocks.8.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 602 | `blocks.8.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 603 | `blocks.8.modulation` | [9216] | BF16 | 9,216 |
| 604 | `blocks.8.norm2.bias` | [1536] | BF16 | 1,536 |
| 605 | `blocks.8.norm2.weight` | [1536] | BF16 | 1,536 |
| 606 | `blocks.8.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 607 | `blocks.8.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 608 | `blocks.8.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 609 | `blocks.8.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 610 | `blocks.8.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 611 | `blocks.8.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 612 | `blocks.9.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 613 | `blocks.9.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 614 | `blocks.9.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 615 | `blocks.9.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 616 | `blocks.9.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 617 | `blocks.9.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 618 | `blocks.9.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 619 | `blocks.9.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 620 | `blocks.9.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 621 | `blocks.9.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 622 | `blocks.9.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 623 | `blocks.9.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 624 | `blocks.9.modulation` | [9216] | BF16 | 9,216 |
| 625 | `blocks.9.norm2.bias` | [1536] | BF16 | 1,536 |
| 626 | `blocks.9.norm2.weight` | [1536] | BF16 | 1,536 |
| 627 | `blocks.9.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 628 | `blocks.9.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 629 | `blocks.9.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 630 | `blocks.9.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 631 | `blocks.9.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 632 | `blocks.9.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 633 | `input_layer.bias` | [1536] | BF16 | 1,536 |
| 634 | `input_layer.weight` | [1536, 64] | BF16 | 98,304 |
| 635 | `out_layer.bias` | [32] | BF16 | 32 |
| 636 | `out_layer.weight` | [32, 1536] | BF16 | 49,152 |
| 637 | `t_embedder.mlp.0.bias` | [1536] | BF16 | 1,536 |
| 638 | `t_embedder.mlp.0.weight` | [1536, 256] | BF16 | 393,216 |
| 639 | `t_embedder.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 640 | `t_embedder.mlp.2.weight` | [1536, 1536] | BF16 | 2,359,296 |

## `slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors`

**Role:** Stage 3 — Texture SLAT DiT (1024 variant)

Sparse 64³ × 64ch material flow model fine-tuned from the 512 variant.

Total: **640 parameters**, 1292.30M elements.

Top-level prefixes: `adaLN_modulation` (2), `blocks` (630), `input_layer` (2), `out_layer` (2), `t_embedder` (4)

| # | Parameter | Shape | Dtype | Elements |
|---:|---|---|:---:|---:|
| 1 | `adaLN_modulation.1.bias` | [9216] | BF16 | 9,216 |
| 2 | `adaLN_modulation.1.weight` | [9216, 1536] | BF16 | 14,155,776 |
| 3 | `blocks.0.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 4 | `blocks.0.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 5 | `blocks.0.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 6 | `blocks.0.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 7 | `blocks.0.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 8 | `blocks.0.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 9 | `blocks.0.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 10 | `blocks.0.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 11 | `blocks.0.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 12 | `blocks.0.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 13 | `blocks.0.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 14 | `blocks.0.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 15 | `blocks.0.modulation` | [9216] | BF16 | 9,216 |
| 16 | `blocks.0.norm2.bias` | [1536] | BF16 | 1,536 |
| 17 | `blocks.0.norm2.weight` | [1536] | BF16 | 1,536 |
| 18 | `blocks.0.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 19 | `blocks.0.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 20 | `blocks.0.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 21 | `blocks.0.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 22 | `blocks.0.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 23 | `blocks.0.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 24 | `blocks.1.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 25 | `blocks.1.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 26 | `blocks.1.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 27 | `blocks.1.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 28 | `blocks.1.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 29 | `blocks.1.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 30 | `blocks.1.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 31 | `blocks.1.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 32 | `blocks.1.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 33 | `blocks.1.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 34 | `blocks.1.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 35 | `blocks.1.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 36 | `blocks.1.modulation` | [9216] | BF16 | 9,216 |
| 37 | `blocks.1.norm2.bias` | [1536] | BF16 | 1,536 |
| 38 | `blocks.1.norm2.weight` | [1536] | BF16 | 1,536 |
| 39 | `blocks.1.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 40 | `blocks.1.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 41 | `blocks.1.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 42 | `blocks.1.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 43 | `blocks.1.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 44 | `blocks.1.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 45 | `blocks.10.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 46 | `blocks.10.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 47 | `blocks.10.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 48 | `blocks.10.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 49 | `blocks.10.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 50 | `blocks.10.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 51 | `blocks.10.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 52 | `blocks.10.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 53 | `blocks.10.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 54 | `blocks.10.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 55 | `blocks.10.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 56 | `blocks.10.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 57 | `blocks.10.modulation` | [9216] | BF16 | 9,216 |
| 58 | `blocks.10.norm2.bias` | [1536] | BF16 | 1,536 |
| 59 | `blocks.10.norm2.weight` | [1536] | BF16 | 1,536 |
| 60 | `blocks.10.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 61 | `blocks.10.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 62 | `blocks.10.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 63 | `blocks.10.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 64 | `blocks.10.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 65 | `blocks.10.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 66 | `blocks.11.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 67 | `blocks.11.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 68 | `blocks.11.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 69 | `blocks.11.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 70 | `blocks.11.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 71 | `blocks.11.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 72 | `blocks.11.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 73 | `blocks.11.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 74 | `blocks.11.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 75 | `blocks.11.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 76 | `blocks.11.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 77 | `blocks.11.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 78 | `blocks.11.modulation` | [9216] | BF16 | 9,216 |
| 79 | `blocks.11.norm2.bias` | [1536] | BF16 | 1,536 |
| 80 | `blocks.11.norm2.weight` | [1536] | BF16 | 1,536 |
| 81 | `blocks.11.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 82 | `blocks.11.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 83 | `blocks.11.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 84 | `blocks.11.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 85 | `blocks.11.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 86 | `blocks.11.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 87 | `blocks.12.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 88 | `blocks.12.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 89 | `blocks.12.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 90 | `blocks.12.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 91 | `blocks.12.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 92 | `blocks.12.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 93 | `blocks.12.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 94 | `blocks.12.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 95 | `blocks.12.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 96 | `blocks.12.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 97 | `blocks.12.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 98 | `blocks.12.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 99 | `blocks.12.modulation` | [9216] | BF16 | 9,216 |
| 100 | `blocks.12.norm2.bias` | [1536] | BF16 | 1,536 |
| 101 | `blocks.12.norm2.weight` | [1536] | BF16 | 1,536 |
| 102 | `blocks.12.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 103 | `blocks.12.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 104 | `blocks.12.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 105 | `blocks.12.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 106 | `blocks.12.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 107 | `blocks.12.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 108 | `blocks.13.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 109 | `blocks.13.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 110 | `blocks.13.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 111 | `blocks.13.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 112 | `blocks.13.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 113 | `blocks.13.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 114 | `blocks.13.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 115 | `blocks.13.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 116 | `blocks.13.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 117 | `blocks.13.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 118 | `blocks.13.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 119 | `blocks.13.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 120 | `blocks.13.modulation` | [9216] | BF16 | 9,216 |
| 121 | `blocks.13.norm2.bias` | [1536] | BF16 | 1,536 |
| 122 | `blocks.13.norm2.weight` | [1536] | BF16 | 1,536 |
| 123 | `blocks.13.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 124 | `blocks.13.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 125 | `blocks.13.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 126 | `blocks.13.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 127 | `blocks.13.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 128 | `blocks.13.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 129 | `blocks.14.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 130 | `blocks.14.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 131 | `blocks.14.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 132 | `blocks.14.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 133 | `blocks.14.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 134 | `blocks.14.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 135 | `blocks.14.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 136 | `blocks.14.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 137 | `blocks.14.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 138 | `blocks.14.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 139 | `blocks.14.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 140 | `blocks.14.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 141 | `blocks.14.modulation` | [9216] | BF16 | 9,216 |
| 142 | `blocks.14.norm2.bias` | [1536] | BF16 | 1,536 |
| 143 | `blocks.14.norm2.weight` | [1536] | BF16 | 1,536 |
| 144 | `blocks.14.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 145 | `blocks.14.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 146 | `blocks.14.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 147 | `blocks.14.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 148 | `blocks.14.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 149 | `blocks.14.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 150 | `blocks.15.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 151 | `blocks.15.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 152 | `blocks.15.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 153 | `blocks.15.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 154 | `blocks.15.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 155 | `blocks.15.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 156 | `blocks.15.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 157 | `blocks.15.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 158 | `blocks.15.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 159 | `blocks.15.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 160 | `blocks.15.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 161 | `blocks.15.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 162 | `blocks.15.modulation` | [9216] | BF16 | 9,216 |
| 163 | `blocks.15.norm2.bias` | [1536] | BF16 | 1,536 |
| 164 | `blocks.15.norm2.weight` | [1536] | BF16 | 1,536 |
| 165 | `blocks.15.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 166 | `blocks.15.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 167 | `blocks.15.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 168 | `blocks.15.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 169 | `blocks.15.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 170 | `blocks.15.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 171 | `blocks.16.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 172 | `blocks.16.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 173 | `blocks.16.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 174 | `blocks.16.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 175 | `blocks.16.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 176 | `blocks.16.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 177 | `blocks.16.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 178 | `blocks.16.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 179 | `blocks.16.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 180 | `blocks.16.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 181 | `blocks.16.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 182 | `blocks.16.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 183 | `blocks.16.modulation` | [9216] | BF16 | 9,216 |
| 184 | `blocks.16.norm2.bias` | [1536] | BF16 | 1,536 |
| 185 | `blocks.16.norm2.weight` | [1536] | BF16 | 1,536 |
| 186 | `blocks.16.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 187 | `blocks.16.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 188 | `blocks.16.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 189 | `blocks.16.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 190 | `blocks.16.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 191 | `blocks.16.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 192 | `blocks.17.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 193 | `blocks.17.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 194 | `blocks.17.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 195 | `blocks.17.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 196 | `blocks.17.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 197 | `blocks.17.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 198 | `blocks.17.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 199 | `blocks.17.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 200 | `blocks.17.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 201 | `blocks.17.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 202 | `blocks.17.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 203 | `blocks.17.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 204 | `blocks.17.modulation` | [9216] | BF16 | 9,216 |
| 205 | `blocks.17.norm2.bias` | [1536] | BF16 | 1,536 |
| 206 | `blocks.17.norm2.weight` | [1536] | BF16 | 1,536 |
| 207 | `blocks.17.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 208 | `blocks.17.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 209 | `blocks.17.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 210 | `blocks.17.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 211 | `blocks.17.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 212 | `blocks.17.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 213 | `blocks.18.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 214 | `blocks.18.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 215 | `blocks.18.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 216 | `blocks.18.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 217 | `blocks.18.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 218 | `blocks.18.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 219 | `blocks.18.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 220 | `blocks.18.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 221 | `blocks.18.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 222 | `blocks.18.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 223 | `blocks.18.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 224 | `blocks.18.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 225 | `blocks.18.modulation` | [9216] | BF16 | 9,216 |
| 226 | `blocks.18.norm2.bias` | [1536] | BF16 | 1,536 |
| 227 | `blocks.18.norm2.weight` | [1536] | BF16 | 1,536 |
| 228 | `blocks.18.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 229 | `blocks.18.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 230 | `blocks.18.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 231 | `blocks.18.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 232 | `blocks.18.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 233 | `blocks.18.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 234 | `blocks.19.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 235 | `blocks.19.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 236 | `blocks.19.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 237 | `blocks.19.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 238 | `blocks.19.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 239 | `blocks.19.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 240 | `blocks.19.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 241 | `blocks.19.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 242 | `blocks.19.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 243 | `blocks.19.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 244 | `blocks.19.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 245 | `blocks.19.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 246 | `blocks.19.modulation` | [9216] | BF16 | 9,216 |
| 247 | `blocks.19.norm2.bias` | [1536] | BF16 | 1,536 |
| 248 | `blocks.19.norm2.weight` | [1536] | BF16 | 1,536 |
| 249 | `blocks.19.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 250 | `blocks.19.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 251 | `blocks.19.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 252 | `blocks.19.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 253 | `blocks.19.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 254 | `blocks.19.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 255 | `blocks.2.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 256 | `blocks.2.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 257 | `blocks.2.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 258 | `blocks.2.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 259 | `blocks.2.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 260 | `blocks.2.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 261 | `blocks.2.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 262 | `blocks.2.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 263 | `blocks.2.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 264 | `blocks.2.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 265 | `blocks.2.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 266 | `blocks.2.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 267 | `blocks.2.modulation` | [9216] | BF16 | 9,216 |
| 268 | `blocks.2.norm2.bias` | [1536] | BF16 | 1,536 |
| 269 | `blocks.2.norm2.weight` | [1536] | BF16 | 1,536 |
| 270 | `blocks.2.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 271 | `blocks.2.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 272 | `blocks.2.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 273 | `blocks.2.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 274 | `blocks.2.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 275 | `blocks.2.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 276 | `blocks.20.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 277 | `blocks.20.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 278 | `blocks.20.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 279 | `blocks.20.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 280 | `blocks.20.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 281 | `blocks.20.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 282 | `blocks.20.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 283 | `blocks.20.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 284 | `blocks.20.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 285 | `blocks.20.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 286 | `blocks.20.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 287 | `blocks.20.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 288 | `blocks.20.modulation` | [9216] | BF16 | 9,216 |
| 289 | `blocks.20.norm2.bias` | [1536] | BF16 | 1,536 |
| 290 | `blocks.20.norm2.weight` | [1536] | BF16 | 1,536 |
| 291 | `blocks.20.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 292 | `blocks.20.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 293 | `blocks.20.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 294 | `blocks.20.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 295 | `blocks.20.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 296 | `blocks.20.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 297 | `blocks.21.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 298 | `blocks.21.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 299 | `blocks.21.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 300 | `blocks.21.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 301 | `blocks.21.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 302 | `blocks.21.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 303 | `blocks.21.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 304 | `blocks.21.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 305 | `blocks.21.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 306 | `blocks.21.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 307 | `blocks.21.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 308 | `blocks.21.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 309 | `blocks.21.modulation` | [9216] | BF16 | 9,216 |
| 310 | `blocks.21.norm2.bias` | [1536] | BF16 | 1,536 |
| 311 | `blocks.21.norm2.weight` | [1536] | BF16 | 1,536 |
| 312 | `blocks.21.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 313 | `blocks.21.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 314 | `blocks.21.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 315 | `blocks.21.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 316 | `blocks.21.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 317 | `blocks.21.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 318 | `blocks.22.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 319 | `blocks.22.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 320 | `blocks.22.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 321 | `blocks.22.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 322 | `blocks.22.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 323 | `blocks.22.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 324 | `blocks.22.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 325 | `blocks.22.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 326 | `blocks.22.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 327 | `blocks.22.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 328 | `blocks.22.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 329 | `blocks.22.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 330 | `blocks.22.modulation` | [9216] | BF16 | 9,216 |
| 331 | `blocks.22.norm2.bias` | [1536] | BF16 | 1,536 |
| 332 | `blocks.22.norm2.weight` | [1536] | BF16 | 1,536 |
| 333 | `blocks.22.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 334 | `blocks.22.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 335 | `blocks.22.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 336 | `blocks.22.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 337 | `blocks.22.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 338 | `blocks.22.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 339 | `blocks.23.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 340 | `blocks.23.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 341 | `blocks.23.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 342 | `blocks.23.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 343 | `blocks.23.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 344 | `blocks.23.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 345 | `blocks.23.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 346 | `blocks.23.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 347 | `blocks.23.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 348 | `blocks.23.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 349 | `blocks.23.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 350 | `blocks.23.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 351 | `blocks.23.modulation` | [9216] | BF16 | 9,216 |
| 352 | `blocks.23.norm2.bias` | [1536] | BF16 | 1,536 |
| 353 | `blocks.23.norm2.weight` | [1536] | BF16 | 1,536 |
| 354 | `blocks.23.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 355 | `blocks.23.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 356 | `blocks.23.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 357 | `blocks.23.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 358 | `blocks.23.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 359 | `blocks.23.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 360 | `blocks.24.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 361 | `blocks.24.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 362 | `blocks.24.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 363 | `blocks.24.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 364 | `blocks.24.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 365 | `blocks.24.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 366 | `blocks.24.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 367 | `blocks.24.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 368 | `blocks.24.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 369 | `blocks.24.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 370 | `blocks.24.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 371 | `blocks.24.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 372 | `blocks.24.modulation` | [9216] | BF16 | 9,216 |
| 373 | `blocks.24.norm2.bias` | [1536] | BF16 | 1,536 |
| 374 | `blocks.24.norm2.weight` | [1536] | BF16 | 1,536 |
| 375 | `blocks.24.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 376 | `blocks.24.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 377 | `blocks.24.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 378 | `blocks.24.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 379 | `blocks.24.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 380 | `blocks.24.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 381 | `blocks.25.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 382 | `blocks.25.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 383 | `blocks.25.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 384 | `blocks.25.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 385 | `blocks.25.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 386 | `blocks.25.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 387 | `blocks.25.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 388 | `blocks.25.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 389 | `blocks.25.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 390 | `blocks.25.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 391 | `blocks.25.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 392 | `blocks.25.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 393 | `blocks.25.modulation` | [9216] | BF16 | 9,216 |
| 394 | `blocks.25.norm2.bias` | [1536] | BF16 | 1,536 |
| 395 | `blocks.25.norm2.weight` | [1536] | BF16 | 1,536 |
| 396 | `blocks.25.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 397 | `blocks.25.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 398 | `blocks.25.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 399 | `blocks.25.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 400 | `blocks.25.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 401 | `blocks.25.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 402 | `blocks.26.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 403 | `blocks.26.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 404 | `blocks.26.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 405 | `blocks.26.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 406 | `blocks.26.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 407 | `blocks.26.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 408 | `blocks.26.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 409 | `blocks.26.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 410 | `blocks.26.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 411 | `blocks.26.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 412 | `blocks.26.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 413 | `blocks.26.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 414 | `blocks.26.modulation` | [9216] | BF16 | 9,216 |
| 415 | `blocks.26.norm2.bias` | [1536] | BF16 | 1,536 |
| 416 | `blocks.26.norm2.weight` | [1536] | BF16 | 1,536 |
| 417 | `blocks.26.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 418 | `blocks.26.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 419 | `blocks.26.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 420 | `blocks.26.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 421 | `blocks.26.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 422 | `blocks.26.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 423 | `blocks.27.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 424 | `blocks.27.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 425 | `blocks.27.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 426 | `blocks.27.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 427 | `blocks.27.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 428 | `blocks.27.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 429 | `blocks.27.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 430 | `blocks.27.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 431 | `blocks.27.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 432 | `blocks.27.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 433 | `blocks.27.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 434 | `blocks.27.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 435 | `blocks.27.modulation` | [9216] | BF16 | 9,216 |
| 436 | `blocks.27.norm2.bias` | [1536] | BF16 | 1,536 |
| 437 | `blocks.27.norm2.weight` | [1536] | BF16 | 1,536 |
| 438 | `blocks.27.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 439 | `blocks.27.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 440 | `blocks.27.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 441 | `blocks.27.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 442 | `blocks.27.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 443 | `blocks.27.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 444 | `blocks.28.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 445 | `blocks.28.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 446 | `blocks.28.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 447 | `blocks.28.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 448 | `blocks.28.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 449 | `blocks.28.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 450 | `blocks.28.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 451 | `blocks.28.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 452 | `blocks.28.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 453 | `blocks.28.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 454 | `blocks.28.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 455 | `blocks.28.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 456 | `blocks.28.modulation` | [9216] | BF16 | 9,216 |
| 457 | `blocks.28.norm2.bias` | [1536] | BF16 | 1,536 |
| 458 | `blocks.28.norm2.weight` | [1536] | BF16 | 1,536 |
| 459 | `blocks.28.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 460 | `blocks.28.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 461 | `blocks.28.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 462 | `blocks.28.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 463 | `blocks.28.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 464 | `blocks.28.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 465 | `blocks.29.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 466 | `blocks.29.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 467 | `blocks.29.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 468 | `blocks.29.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 469 | `blocks.29.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 470 | `blocks.29.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 471 | `blocks.29.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 472 | `blocks.29.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 473 | `blocks.29.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 474 | `blocks.29.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 475 | `blocks.29.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 476 | `blocks.29.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 477 | `blocks.29.modulation` | [9216] | BF16 | 9,216 |
| 478 | `blocks.29.norm2.bias` | [1536] | BF16 | 1,536 |
| 479 | `blocks.29.norm2.weight` | [1536] | BF16 | 1,536 |
| 480 | `blocks.29.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 481 | `blocks.29.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 482 | `blocks.29.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 483 | `blocks.29.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 484 | `blocks.29.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 485 | `blocks.29.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 486 | `blocks.3.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 487 | `blocks.3.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 488 | `blocks.3.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 489 | `blocks.3.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 490 | `blocks.3.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 491 | `blocks.3.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 492 | `blocks.3.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 493 | `blocks.3.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 494 | `blocks.3.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 495 | `blocks.3.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 496 | `blocks.3.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 497 | `blocks.3.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 498 | `blocks.3.modulation` | [9216] | BF16 | 9,216 |
| 499 | `blocks.3.norm2.bias` | [1536] | BF16 | 1,536 |
| 500 | `blocks.3.norm2.weight` | [1536] | BF16 | 1,536 |
| 501 | `blocks.3.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 502 | `blocks.3.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 503 | `blocks.3.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 504 | `blocks.3.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 505 | `blocks.3.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 506 | `blocks.3.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 507 | `blocks.4.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 508 | `blocks.4.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 509 | `blocks.4.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 510 | `blocks.4.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 511 | `blocks.4.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 512 | `blocks.4.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 513 | `blocks.4.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 514 | `blocks.4.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 515 | `blocks.4.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 516 | `blocks.4.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 517 | `blocks.4.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 518 | `blocks.4.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 519 | `blocks.4.modulation` | [9216] | BF16 | 9,216 |
| 520 | `blocks.4.norm2.bias` | [1536] | BF16 | 1,536 |
| 521 | `blocks.4.norm2.weight` | [1536] | BF16 | 1,536 |
| 522 | `blocks.4.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 523 | `blocks.4.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 524 | `blocks.4.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 525 | `blocks.4.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 526 | `blocks.4.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 527 | `blocks.4.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 528 | `blocks.5.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 529 | `blocks.5.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 530 | `blocks.5.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 531 | `blocks.5.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 532 | `blocks.5.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 533 | `blocks.5.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 534 | `blocks.5.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 535 | `blocks.5.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 536 | `blocks.5.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 537 | `blocks.5.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 538 | `blocks.5.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 539 | `blocks.5.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 540 | `blocks.5.modulation` | [9216] | BF16 | 9,216 |
| 541 | `blocks.5.norm2.bias` | [1536] | BF16 | 1,536 |
| 542 | `blocks.5.norm2.weight` | [1536] | BF16 | 1,536 |
| 543 | `blocks.5.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 544 | `blocks.5.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 545 | `blocks.5.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 546 | `blocks.5.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 547 | `blocks.5.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 548 | `blocks.5.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 549 | `blocks.6.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 550 | `blocks.6.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 551 | `blocks.6.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 552 | `blocks.6.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 553 | `blocks.6.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 554 | `blocks.6.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 555 | `blocks.6.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 556 | `blocks.6.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 557 | `blocks.6.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 558 | `blocks.6.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 559 | `blocks.6.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 560 | `blocks.6.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 561 | `blocks.6.modulation` | [9216] | BF16 | 9,216 |
| 562 | `blocks.6.norm2.bias` | [1536] | BF16 | 1,536 |
| 563 | `blocks.6.norm2.weight` | [1536] | BF16 | 1,536 |
| 564 | `blocks.6.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 565 | `blocks.6.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 566 | `blocks.6.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 567 | `blocks.6.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 568 | `blocks.6.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 569 | `blocks.6.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 570 | `blocks.7.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 571 | `blocks.7.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 572 | `blocks.7.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 573 | `blocks.7.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 574 | `blocks.7.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 575 | `blocks.7.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 576 | `blocks.7.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 577 | `blocks.7.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 578 | `blocks.7.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 579 | `blocks.7.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 580 | `blocks.7.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 581 | `blocks.7.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 582 | `blocks.7.modulation` | [9216] | BF16 | 9,216 |
| 583 | `blocks.7.norm2.bias` | [1536] | BF16 | 1,536 |
| 584 | `blocks.7.norm2.weight` | [1536] | BF16 | 1,536 |
| 585 | `blocks.7.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 586 | `blocks.7.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 587 | `blocks.7.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 588 | `blocks.7.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 589 | `blocks.7.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 590 | `blocks.7.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 591 | `blocks.8.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 592 | `blocks.8.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 593 | `blocks.8.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 594 | `blocks.8.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 595 | `blocks.8.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 596 | `blocks.8.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 597 | `blocks.8.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 598 | `blocks.8.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 599 | `blocks.8.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 600 | `blocks.8.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 601 | `blocks.8.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 602 | `blocks.8.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 603 | `blocks.8.modulation` | [9216] | BF16 | 9,216 |
| 604 | `blocks.8.norm2.bias` | [1536] | BF16 | 1,536 |
| 605 | `blocks.8.norm2.weight` | [1536] | BF16 | 1,536 |
| 606 | `blocks.8.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 607 | `blocks.8.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 608 | `blocks.8.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 609 | `blocks.8.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 610 | `blocks.8.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 611 | `blocks.8.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 612 | `blocks.9.cross_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 613 | `blocks.9.cross_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 614 | `blocks.9.cross_attn.to_kv.bias` | [3072] | BF16 | 3,072 |
| 615 | `blocks.9.cross_attn.to_kv.weight` | [3072, 1024] | BF16 | 3,145,728 |
| 616 | `blocks.9.cross_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 617 | `blocks.9.cross_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 618 | `blocks.9.cross_attn.to_q.bias` | [1536] | BF16 | 1,536 |
| 619 | `blocks.9.cross_attn.to_q.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 620 | `blocks.9.mlp.mlp.0.bias` | [8192] | BF16 | 8,192 |
| 621 | `blocks.9.mlp.mlp.0.weight` | [8192, 1536] | BF16 | 12,582,912 |
| 622 | `blocks.9.mlp.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 623 | `blocks.9.mlp.mlp.2.weight` | [1536, 8192] | BF16 | 12,582,912 |
| 624 | `blocks.9.modulation` | [9216] | BF16 | 9,216 |
| 625 | `blocks.9.norm2.bias` | [1536] | BF16 | 1,536 |
| 626 | `blocks.9.norm2.weight` | [1536] | BF16 | 1,536 |
| 627 | `blocks.9.self_attn.k_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 628 | `blocks.9.self_attn.q_rms_norm.gamma` | [12, 128] | BF16 | 1,536 |
| 629 | `blocks.9.self_attn.to_out.bias` | [1536] | BF16 | 1,536 |
| 630 | `blocks.9.self_attn.to_out.weight` | [1536, 1536] | BF16 | 2,359,296 |
| 631 | `blocks.9.self_attn.to_qkv.bias` | [4608] | BF16 | 4,608 |
| 632 | `blocks.9.self_attn.to_qkv.weight` | [4608, 1536] | BF16 | 7,077,888 |
| 633 | `input_layer.bias` | [1536] | BF16 | 1,536 |
| 634 | `input_layer.weight` | [1536, 64] | BF16 | 98,304 |
| 635 | `out_layer.bias` | [32] | BF16 | 32 |
| 636 | `out_layer.weight` | [32, 1536] | BF16 | 49,152 |
| 637 | `t_embedder.mlp.0.bias` | [1536] | BF16 | 1,536 |
| 638 | `t_embedder.mlp.0.weight` | [1536, 256] | BF16 | 393,216 |
| 639 | `t_embedder.mlp.2.bias` | [1536] | BF16 | 1,536 |
| 640 | `t_embedder.mlp.2.weight` | [1536, 1536] | BF16 | 2,359,296 |

## `shape_enc_next_dc_f16c32_fp16.safetensors`

**Role:** SC-VAE shape encoder

FlexiDualGridVaeEncoder: 5-stage sparse-conv UNet encoder, in_channels=6 (vertex offset 3 + intersection flags 3), latent 32 channels. Used for training; not required for inference.

Total: **284 parameters**, 354.39M elements.

Top-level prefixes: `blocks` (280), `input_layer` (2), `to_latent` (2)

| # | Parameter | Shape | Dtype | Elements |
|---:|---|---|:---:|---:|
| 1 | `blocks.0.0.conv1.bias` | [16] | F16 | 16 |
| 2 | `blocks.0.0.conv1.weight` | [16, 3, 3, 3, 64] | F16 | 27,648 |
| 3 | `blocks.0.0.conv2.bias` | [128] | F16 | 128 |
| 4 | `blocks.0.0.conv2.weight` | [128, 3, 3, 3, 128] | F16 | 442,368 |
| 5 | `blocks.0.0.norm1.bias` | [64] | F16 | 64 |
| 6 | `blocks.0.0.norm1.weight` | [64] | F16 | 64 |
| 7 | `blocks.1.0.conv.bias` | [128] | F16 | 128 |
| 8 | `blocks.1.0.conv.weight` | [128, 3, 3, 3, 128] | F16 | 442,368 |
| 9 | `blocks.1.0.mlp.0.bias` | [512] | F16 | 512 |
| 10 | `blocks.1.0.mlp.0.weight` | [512, 128] | F16 | 65,536 |
| 11 | `blocks.1.0.mlp.2.bias` | [128] | F16 | 128 |
| 12 | `blocks.1.0.mlp.2.weight` | [128, 512] | F16 | 65,536 |
| 13 | `blocks.1.0.norm.bias` | [128] | F16 | 128 |
| 14 | `blocks.1.0.norm.weight` | [128] | F16 | 128 |
| 15 | `blocks.1.1.conv.bias` | [128] | F16 | 128 |
| 16 | `blocks.1.1.conv.weight` | [128, 3, 3, 3, 128] | F16 | 442,368 |
| 17 | `blocks.1.1.mlp.0.bias` | [512] | F16 | 512 |
| 18 | `blocks.1.1.mlp.0.weight` | [512, 128] | F16 | 65,536 |
| 19 | `blocks.1.1.mlp.2.bias` | [128] | F16 | 128 |
| 20 | `blocks.1.1.mlp.2.weight` | [128, 512] | F16 | 65,536 |
| 21 | `blocks.1.1.norm.bias` | [128] | F16 | 128 |
| 22 | `blocks.1.1.norm.weight` | [128] | F16 | 128 |
| 23 | `blocks.1.2.conv.bias` | [128] | F16 | 128 |
| 24 | `blocks.1.2.conv.weight` | [128, 3, 3, 3, 128] | F16 | 442,368 |
| 25 | `blocks.1.2.mlp.0.bias` | [512] | F16 | 512 |
| 26 | `blocks.1.2.mlp.0.weight` | [512, 128] | F16 | 65,536 |
| 27 | `blocks.1.2.mlp.2.bias` | [128] | F16 | 128 |
| 28 | `blocks.1.2.mlp.2.weight` | [128, 512] | F16 | 65,536 |
| 29 | `blocks.1.2.norm.bias` | [128] | F16 | 128 |
| 30 | `blocks.1.2.norm.weight` | [128] | F16 | 128 |
| 31 | `blocks.1.3.conv.bias` | [128] | F16 | 128 |
| 32 | `blocks.1.3.conv.weight` | [128, 3, 3, 3, 128] | F16 | 442,368 |
| 33 | `blocks.1.3.mlp.0.bias` | [512] | F16 | 512 |
| 34 | `blocks.1.3.mlp.0.weight` | [512, 128] | F16 | 65,536 |
| 35 | `blocks.1.3.mlp.2.bias` | [128] | F16 | 128 |
| 36 | `blocks.1.3.mlp.2.weight` | [128, 512] | F16 | 65,536 |
| 37 | `blocks.1.3.norm.bias` | [128] | F16 | 128 |
| 38 | `blocks.1.3.norm.weight` | [128] | F16 | 128 |
| 39 | `blocks.1.4.conv1.bias` | [32] | F16 | 32 |
| 40 | `blocks.1.4.conv1.weight` | [32, 3, 3, 3, 128] | F16 | 110,592 |
| 41 | `blocks.1.4.conv2.bias` | [256] | F16 | 256 |
| 42 | `blocks.1.4.conv2.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 43 | `blocks.1.4.norm1.bias` | [128] | F16 | 128 |
| 44 | `blocks.1.4.norm1.weight` | [128] | F16 | 128 |
| 45 | `blocks.2.0.conv.bias` | [256] | F16 | 256 |
| 46 | `blocks.2.0.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 47 | `blocks.2.0.mlp.0.bias` | [1024] | F16 | 1,024 |
| 48 | `blocks.2.0.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 49 | `blocks.2.0.mlp.2.bias` | [256] | F16 | 256 |
| 50 | `blocks.2.0.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 51 | `blocks.2.0.norm.bias` | [256] | F16 | 256 |
| 52 | `blocks.2.0.norm.weight` | [256] | F16 | 256 |
| 53 | `blocks.2.1.conv.bias` | [256] | F16 | 256 |
| 54 | `blocks.2.1.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 55 | `blocks.2.1.mlp.0.bias` | [1024] | F16 | 1,024 |
| 56 | `blocks.2.1.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 57 | `blocks.2.1.mlp.2.bias` | [256] | F16 | 256 |
| 58 | `blocks.2.1.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 59 | `blocks.2.1.norm.bias` | [256] | F16 | 256 |
| 60 | `blocks.2.1.norm.weight` | [256] | F16 | 256 |
| 61 | `blocks.2.2.conv.bias` | [256] | F16 | 256 |
| 62 | `blocks.2.2.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 63 | `blocks.2.2.mlp.0.bias` | [1024] | F16 | 1,024 |
| 64 | `blocks.2.2.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 65 | `blocks.2.2.mlp.2.bias` | [256] | F16 | 256 |
| 66 | `blocks.2.2.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 67 | `blocks.2.2.norm.bias` | [256] | F16 | 256 |
| 68 | `blocks.2.2.norm.weight` | [256] | F16 | 256 |
| 69 | `blocks.2.3.conv.bias` | [256] | F16 | 256 |
| 70 | `blocks.2.3.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 71 | `blocks.2.3.mlp.0.bias` | [1024] | F16 | 1,024 |
| 72 | `blocks.2.3.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 73 | `blocks.2.3.mlp.2.bias` | [256] | F16 | 256 |
| 74 | `blocks.2.3.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 75 | `blocks.2.3.norm.bias` | [256] | F16 | 256 |
| 76 | `blocks.2.3.norm.weight` | [256] | F16 | 256 |
| 77 | `blocks.2.4.conv.bias` | [256] | F16 | 256 |
| 78 | `blocks.2.4.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 79 | `blocks.2.4.mlp.0.bias` | [1024] | F16 | 1,024 |
| 80 | `blocks.2.4.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 81 | `blocks.2.4.mlp.2.bias` | [256] | F16 | 256 |
| 82 | `blocks.2.4.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 83 | `blocks.2.4.norm.bias` | [256] | F16 | 256 |
| 84 | `blocks.2.4.norm.weight` | [256] | F16 | 256 |
| 85 | `blocks.2.5.conv.bias` | [256] | F16 | 256 |
| 86 | `blocks.2.5.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 87 | `blocks.2.5.mlp.0.bias` | [1024] | F16 | 1,024 |
| 88 | `blocks.2.5.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 89 | `blocks.2.5.mlp.2.bias` | [256] | F16 | 256 |
| 90 | `blocks.2.5.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 91 | `blocks.2.5.norm.bias` | [256] | F16 | 256 |
| 92 | `blocks.2.5.norm.weight` | [256] | F16 | 256 |
| 93 | `blocks.2.6.conv.bias` | [256] | F16 | 256 |
| 94 | `blocks.2.6.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 95 | `blocks.2.6.mlp.0.bias` | [1024] | F16 | 1,024 |
| 96 | `blocks.2.6.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 97 | `blocks.2.6.mlp.2.bias` | [256] | F16 | 256 |
| 98 | `blocks.2.6.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 99 | `blocks.2.6.norm.bias` | [256] | F16 | 256 |
| 100 | `blocks.2.6.norm.weight` | [256] | F16 | 256 |
| 101 | `blocks.2.7.conv.bias` | [256] | F16 | 256 |
| 102 | `blocks.2.7.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 103 | `blocks.2.7.mlp.0.bias` | [1024] | F16 | 1,024 |
| 104 | `blocks.2.7.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 105 | `blocks.2.7.mlp.2.bias` | [256] | F16 | 256 |
| 106 | `blocks.2.7.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 107 | `blocks.2.7.norm.bias` | [256] | F16 | 256 |
| 108 | `blocks.2.7.norm.weight` | [256] | F16 | 256 |
| 109 | `blocks.2.8.conv1.bias` | [64] | F16 | 64 |
| 110 | `blocks.2.8.conv1.weight` | [64, 3, 3, 3, 256] | F16 | 442,368 |
| 111 | `blocks.2.8.conv2.bias` | [512] | F16 | 512 |
| 112 | `blocks.2.8.conv2.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 113 | `blocks.2.8.norm1.bias` | [256] | F16 | 256 |
| 114 | `blocks.2.8.norm1.weight` | [256] | F16 | 256 |
| 115 | `blocks.3.0.conv.bias` | [512] | F16 | 512 |
| 116 | `blocks.3.0.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 117 | `blocks.3.0.mlp.0.bias` | [2048] | F16 | 2,048 |
| 118 | `blocks.3.0.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 119 | `blocks.3.0.mlp.2.bias` | [512] | F16 | 512 |
| 120 | `blocks.3.0.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 121 | `blocks.3.0.norm.bias` | [512] | F16 | 512 |
| 122 | `blocks.3.0.norm.weight` | [512] | F16 | 512 |
| 123 | `blocks.3.1.conv.bias` | [512] | F16 | 512 |
| 124 | `blocks.3.1.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 125 | `blocks.3.1.mlp.0.bias` | [2048] | F16 | 2,048 |
| 126 | `blocks.3.1.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 127 | `blocks.3.1.mlp.2.bias` | [512] | F16 | 512 |
| 128 | `blocks.3.1.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 129 | `blocks.3.1.norm.bias` | [512] | F16 | 512 |
| 130 | `blocks.3.1.norm.weight` | [512] | F16 | 512 |
| 131 | `blocks.3.10.conv.bias` | [512] | F16 | 512 |
| 132 | `blocks.3.10.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 133 | `blocks.3.10.mlp.0.bias` | [2048] | F16 | 2,048 |
| 134 | `blocks.3.10.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 135 | `blocks.3.10.mlp.2.bias` | [512] | F16 | 512 |
| 136 | `blocks.3.10.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 137 | `blocks.3.10.norm.bias` | [512] | F16 | 512 |
| 138 | `blocks.3.10.norm.weight` | [512] | F16 | 512 |
| 139 | `blocks.3.11.conv.bias` | [512] | F16 | 512 |
| 140 | `blocks.3.11.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 141 | `blocks.3.11.mlp.0.bias` | [2048] | F16 | 2,048 |
| 142 | `blocks.3.11.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 143 | `blocks.3.11.mlp.2.bias` | [512] | F16 | 512 |
| 144 | `blocks.3.11.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 145 | `blocks.3.11.norm.bias` | [512] | F16 | 512 |
| 146 | `blocks.3.11.norm.weight` | [512] | F16 | 512 |
| 147 | `blocks.3.12.conv.bias` | [512] | F16 | 512 |
| 148 | `blocks.3.12.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 149 | `blocks.3.12.mlp.0.bias` | [2048] | F16 | 2,048 |
| 150 | `blocks.3.12.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 151 | `blocks.3.12.mlp.2.bias` | [512] | F16 | 512 |
| 152 | `blocks.3.12.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 153 | `blocks.3.12.norm.bias` | [512] | F16 | 512 |
| 154 | `blocks.3.12.norm.weight` | [512] | F16 | 512 |
| 155 | `blocks.3.13.conv.bias` | [512] | F16 | 512 |
| 156 | `blocks.3.13.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 157 | `blocks.3.13.mlp.0.bias` | [2048] | F16 | 2,048 |
| 158 | `blocks.3.13.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 159 | `blocks.3.13.mlp.2.bias` | [512] | F16 | 512 |
| 160 | `blocks.3.13.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 161 | `blocks.3.13.norm.bias` | [512] | F16 | 512 |
| 162 | `blocks.3.13.norm.weight` | [512] | F16 | 512 |
| 163 | `blocks.3.14.conv.bias` | [512] | F16 | 512 |
| 164 | `blocks.3.14.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 165 | `blocks.3.14.mlp.0.bias` | [2048] | F16 | 2,048 |
| 166 | `blocks.3.14.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 167 | `blocks.3.14.mlp.2.bias` | [512] | F16 | 512 |
| 168 | `blocks.3.14.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 169 | `blocks.3.14.norm.bias` | [512] | F16 | 512 |
| 170 | `blocks.3.14.norm.weight` | [512] | F16 | 512 |
| 171 | `blocks.3.15.conv.bias` | [512] | F16 | 512 |
| 172 | `blocks.3.15.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 173 | `blocks.3.15.mlp.0.bias` | [2048] | F16 | 2,048 |
| 174 | `blocks.3.15.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 175 | `blocks.3.15.mlp.2.bias` | [512] | F16 | 512 |
| 176 | `blocks.3.15.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 177 | `blocks.3.15.norm.bias` | [512] | F16 | 512 |
| 178 | `blocks.3.15.norm.weight` | [512] | F16 | 512 |
| 179 | `blocks.3.16.conv1.bias` | [128] | F16 | 128 |
| 180 | `blocks.3.16.conv1.weight` | [128, 3, 3, 3, 512] | F16 | 1,769,472 |
| 181 | `blocks.3.16.conv2.bias` | [1024] | F16 | 1,024 |
| 182 | `blocks.3.16.conv2.weight` | [1024, 3, 3, 3, 1024] | F16 | 28,311,552 |
| 183 | `blocks.3.16.norm1.bias` | [512] | F16 | 512 |
| 184 | `blocks.3.16.norm1.weight` | [512] | F16 | 512 |
| 185 | `blocks.3.2.conv.bias` | [512] | F16 | 512 |
| 186 | `blocks.3.2.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 187 | `blocks.3.2.mlp.0.bias` | [2048] | F16 | 2,048 |
| 188 | `blocks.3.2.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 189 | `blocks.3.2.mlp.2.bias` | [512] | F16 | 512 |
| 190 | `blocks.3.2.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 191 | `blocks.3.2.norm.bias` | [512] | F16 | 512 |
| 192 | `blocks.3.2.norm.weight` | [512] | F16 | 512 |
| 193 | `blocks.3.3.conv.bias` | [512] | F16 | 512 |
| 194 | `blocks.3.3.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 195 | `blocks.3.3.mlp.0.bias` | [2048] | F16 | 2,048 |
| 196 | `blocks.3.3.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 197 | `blocks.3.3.mlp.2.bias` | [512] | F16 | 512 |
| 198 | `blocks.3.3.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 199 | `blocks.3.3.norm.bias` | [512] | F16 | 512 |
| 200 | `blocks.3.3.norm.weight` | [512] | F16 | 512 |
| 201 | `blocks.3.4.conv.bias` | [512] | F16 | 512 |
| 202 | `blocks.3.4.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 203 | `blocks.3.4.mlp.0.bias` | [2048] | F16 | 2,048 |
| 204 | `blocks.3.4.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 205 | `blocks.3.4.mlp.2.bias` | [512] | F16 | 512 |
| 206 | `blocks.3.4.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 207 | `blocks.3.4.norm.bias` | [512] | F16 | 512 |
| 208 | `blocks.3.4.norm.weight` | [512] | F16 | 512 |
| 209 | `blocks.3.5.conv.bias` | [512] | F16 | 512 |
| 210 | `blocks.3.5.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 211 | `blocks.3.5.mlp.0.bias` | [2048] | F16 | 2,048 |
| 212 | `blocks.3.5.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 213 | `blocks.3.5.mlp.2.bias` | [512] | F16 | 512 |
| 214 | `blocks.3.5.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 215 | `blocks.3.5.norm.bias` | [512] | F16 | 512 |
| 216 | `blocks.3.5.norm.weight` | [512] | F16 | 512 |
| 217 | `blocks.3.6.conv.bias` | [512] | F16 | 512 |
| 218 | `blocks.3.6.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 219 | `blocks.3.6.mlp.0.bias` | [2048] | F16 | 2,048 |
| 220 | `blocks.3.6.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 221 | `blocks.3.6.mlp.2.bias` | [512] | F16 | 512 |
| 222 | `blocks.3.6.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 223 | `blocks.3.6.norm.bias` | [512] | F16 | 512 |
| 224 | `blocks.3.6.norm.weight` | [512] | F16 | 512 |
| 225 | `blocks.3.7.conv.bias` | [512] | F16 | 512 |
| 226 | `blocks.3.7.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 227 | `blocks.3.7.mlp.0.bias` | [2048] | F16 | 2,048 |
| 228 | `blocks.3.7.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 229 | `blocks.3.7.mlp.2.bias` | [512] | F16 | 512 |
| 230 | `blocks.3.7.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 231 | `blocks.3.7.norm.bias` | [512] | F16 | 512 |
| 232 | `blocks.3.7.norm.weight` | [512] | F16 | 512 |
| 233 | `blocks.3.8.conv.bias` | [512] | F16 | 512 |
| 234 | `blocks.3.8.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 235 | `blocks.3.8.mlp.0.bias` | [2048] | F16 | 2,048 |
| 236 | `blocks.3.8.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 237 | `blocks.3.8.mlp.2.bias` | [512] | F16 | 512 |
| 238 | `blocks.3.8.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 239 | `blocks.3.8.norm.bias` | [512] | F16 | 512 |
| 240 | `blocks.3.8.norm.weight` | [512] | F16 | 512 |
| 241 | `blocks.3.9.conv.bias` | [512] | F16 | 512 |
| 242 | `blocks.3.9.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 243 | `blocks.3.9.mlp.0.bias` | [2048] | F16 | 2,048 |
| 244 | `blocks.3.9.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 245 | `blocks.3.9.mlp.2.bias` | [512] | F16 | 512 |
| 246 | `blocks.3.9.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 247 | `blocks.3.9.norm.bias` | [512] | F16 | 512 |
| 248 | `blocks.3.9.norm.weight` | [512] | F16 | 512 |
| 249 | `blocks.4.0.conv.bias` | [1024] | F16 | 1,024 |
| 250 | `blocks.4.0.conv.weight` | [1024, 3, 3, 3, 1024] | F16 | 28,311,552 |
| 251 | `blocks.4.0.mlp.0.bias` | [4096] | F16 | 4,096 |
| 252 | `blocks.4.0.mlp.0.weight` | [4096, 1024] | F16 | 4,194,304 |
| 253 | `blocks.4.0.mlp.2.bias` | [1024] | F16 | 1,024 |
| 254 | `blocks.4.0.mlp.2.weight` | [1024, 4096] | F16 | 4,194,304 |
| 255 | `blocks.4.0.norm.bias` | [1024] | F16 | 1,024 |
| 256 | `blocks.4.0.norm.weight` | [1024] | F16 | 1,024 |
| 257 | `blocks.4.1.conv.bias` | [1024] | F16 | 1,024 |
| 258 | `blocks.4.1.conv.weight` | [1024, 3, 3, 3, 1024] | F16 | 28,311,552 |
| 259 | `blocks.4.1.mlp.0.bias` | [4096] | F16 | 4,096 |
| 260 | `blocks.4.1.mlp.0.weight` | [4096, 1024] | F16 | 4,194,304 |
| 261 | `blocks.4.1.mlp.2.bias` | [1024] | F16 | 1,024 |
| 262 | `blocks.4.1.mlp.2.weight` | [1024, 4096] | F16 | 4,194,304 |
| 263 | `blocks.4.1.norm.bias` | [1024] | F16 | 1,024 |
| 264 | `blocks.4.1.norm.weight` | [1024] | F16 | 1,024 |
| 265 | `blocks.4.2.conv.bias` | [1024] | F16 | 1,024 |
| 266 | `blocks.4.2.conv.weight` | [1024, 3, 3, 3, 1024] | F16 | 28,311,552 |
| 267 | `blocks.4.2.mlp.0.bias` | [4096] | F16 | 4,096 |
| 268 | `blocks.4.2.mlp.0.weight` | [4096, 1024] | F16 | 4,194,304 |
| 269 | `blocks.4.2.mlp.2.bias` | [1024] | F16 | 1,024 |
| 270 | `blocks.4.2.mlp.2.weight` | [1024, 4096] | F16 | 4,194,304 |
| 271 | `blocks.4.2.norm.bias` | [1024] | F16 | 1,024 |
| 272 | `blocks.4.2.norm.weight` | [1024] | F16 | 1,024 |
| 273 | `blocks.4.3.conv.bias` | [1024] | F16 | 1,024 |
| 274 | `blocks.4.3.conv.weight` | [1024, 3, 3, 3, 1024] | F16 | 28,311,552 |
| 275 | `blocks.4.3.mlp.0.bias` | [4096] | F16 | 4,096 |
| 276 | `blocks.4.3.mlp.0.weight` | [4096, 1024] | F16 | 4,194,304 |
| 277 | `blocks.4.3.mlp.2.bias` | [1024] | F16 | 1,024 |
| 278 | `blocks.4.3.mlp.2.weight` | [1024, 4096] | F16 | 4,194,304 |
| 279 | `blocks.4.3.norm.bias` | [1024] | F16 | 1,024 |
| 280 | `blocks.4.3.norm.weight` | [1024] | F16 | 1,024 |
| 281 | `input_layer.bias` | [64] | F16 | 64 |
| 282 | `input_layer.weight` | [64, 6] | F16 | 384 |
| 283 | `to_latent.bias` | [64] | F16 | 64 |
| 284 | `to_latent.weight` | [64, 1024] | F16 | 65,536 |

## `shape_dec_next_dc_f16c32_fp16.safetensors`

**Role:** SC-VAE shape decoder

FlexiDualGridVaeDecoder: 5-stage sparse-conv UNet decoder. Per-active-voxel output: (vx, vy, vz, δx, δy, δz, γ) = 7 channels. v is sigmoid in [-0.5, 1.5]; δ is raw logits (threshold 0); γ is softplus in (0,∞).

Total: **292 parameters**, 474.23M elements.

Top-level prefixes: `blocks` (288), `from_latent` (2), `output_layer` (2)

| # | Parameter | Shape | Dtype | Elements |
|---:|---|---|:---:|---:|
| 1 | `blocks.0.0.conv.bias` | [1024] | F16 | 1,024 |
| 2 | `blocks.0.0.conv.weight` | [1024, 3, 3, 3, 1024] | F16 | 28,311,552 |
| 3 | `blocks.0.0.mlp.0.bias` | [4096] | F16 | 4,096 |
| 4 | `blocks.0.0.mlp.0.weight` | [4096, 1024] | F16 | 4,194,304 |
| 5 | `blocks.0.0.mlp.2.bias` | [1024] | F16 | 1,024 |
| 6 | `blocks.0.0.mlp.2.weight` | [1024, 4096] | F16 | 4,194,304 |
| 7 | `blocks.0.0.norm.bias` | [1024] | F16 | 1,024 |
| 8 | `blocks.0.0.norm.weight` | [1024] | F16 | 1,024 |
| 9 | `blocks.0.1.conv.bias` | [1024] | F16 | 1,024 |
| 10 | `blocks.0.1.conv.weight` | [1024, 3, 3, 3, 1024] | F16 | 28,311,552 |
| 11 | `blocks.0.1.mlp.0.bias` | [4096] | F16 | 4,096 |
| 12 | `blocks.0.1.mlp.0.weight` | [4096, 1024] | F16 | 4,194,304 |
| 13 | `blocks.0.1.mlp.2.bias` | [1024] | F16 | 1,024 |
| 14 | `blocks.0.1.mlp.2.weight` | [1024, 4096] | F16 | 4,194,304 |
| 15 | `blocks.0.1.norm.bias` | [1024] | F16 | 1,024 |
| 16 | `blocks.0.1.norm.weight` | [1024] | F16 | 1,024 |
| 17 | `blocks.0.2.conv.bias` | [1024] | F16 | 1,024 |
| 18 | `blocks.0.2.conv.weight` | [1024, 3, 3, 3, 1024] | F16 | 28,311,552 |
| 19 | `blocks.0.2.mlp.0.bias` | [4096] | F16 | 4,096 |
| 20 | `blocks.0.2.mlp.0.weight` | [4096, 1024] | F16 | 4,194,304 |
| 21 | `blocks.0.2.mlp.2.bias` | [1024] | F16 | 1,024 |
| 22 | `blocks.0.2.mlp.2.weight` | [1024, 4096] | F16 | 4,194,304 |
| 23 | `blocks.0.2.norm.bias` | [1024] | F16 | 1,024 |
| 24 | `blocks.0.2.norm.weight` | [1024] | F16 | 1,024 |
| 25 | `blocks.0.3.conv.bias` | [1024] | F16 | 1,024 |
| 26 | `blocks.0.3.conv.weight` | [1024, 3, 3, 3, 1024] | F16 | 28,311,552 |
| 27 | `blocks.0.3.mlp.0.bias` | [4096] | F16 | 4,096 |
| 28 | `blocks.0.3.mlp.0.weight` | [4096, 1024] | F16 | 4,194,304 |
| 29 | `blocks.0.3.mlp.2.bias` | [1024] | F16 | 1,024 |
| 30 | `blocks.0.3.mlp.2.weight` | [1024, 4096] | F16 | 4,194,304 |
| 31 | `blocks.0.3.norm.bias` | [1024] | F16 | 1,024 |
| 32 | `blocks.0.3.norm.weight` | [1024] | F16 | 1,024 |
| 33 | `blocks.0.4.conv1.bias` | [4096] | F16 | 4,096 |
| 34 | `blocks.0.4.conv1.weight` | [4096, 3, 3, 3, 1024] | F16 | 113,246,208 |
| 35 | `blocks.0.4.conv2.bias` | [512] | F16 | 512 |
| 36 | `blocks.0.4.conv2.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 37 | `blocks.0.4.norm1.bias` | [1024] | F16 | 1,024 |
| 38 | `blocks.0.4.norm1.weight` | [1024] | F16 | 1,024 |
| 39 | `blocks.0.4.to_subdiv.bias` | [8] | F16 | 8 |
| 40 | `blocks.0.4.to_subdiv.weight` | [8, 1024] | F16 | 8,192 |
| 41 | `blocks.1.0.conv.bias` | [512] | F16 | 512 |
| 42 | `blocks.1.0.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 43 | `blocks.1.0.mlp.0.bias` | [2048] | F16 | 2,048 |
| 44 | `blocks.1.0.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 45 | `blocks.1.0.mlp.2.bias` | [512] | F16 | 512 |
| 46 | `blocks.1.0.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 47 | `blocks.1.0.norm.bias` | [512] | F16 | 512 |
| 48 | `blocks.1.0.norm.weight` | [512] | F16 | 512 |
| 49 | `blocks.1.1.conv.bias` | [512] | F16 | 512 |
| 50 | `blocks.1.1.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 51 | `blocks.1.1.mlp.0.bias` | [2048] | F16 | 2,048 |
| 52 | `blocks.1.1.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 53 | `blocks.1.1.mlp.2.bias` | [512] | F16 | 512 |
| 54 | `blocks.1.1.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 55 | `blocks.1.1.norm.bias` | [512] | F16 | 512 |
| 56 | `blocks.1.1.norm.weight` | [512] | F16 | 512 |
| 57 | `blocks.1.10.conv.bias` | [512] | F16 | 512 |
| 58 | `blocks.1.10.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 59 | `blocks.1.10.mlp.0.bias` | [2048] | F16 | 2,048 |
| 60 | `blocks.1.10.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 61 | `blocks.1.10.mlp.2.bias` | [512] | F16 | 512 |
| 62 | `blocks.1.10.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 63 | `blocks.1.10.norm.bias` | [512] | F16 | 512 |
| 64 | `blocks.1.10.norm.weight` | [512] | F16 | 512 |
| 65 | `blocks.1.11.conv.bias` | [512] | F16 | 512 |
| 66 | `blocks.1.11.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 67 | `blocks.1.11.mlp.0.bias` | [2048] | F16 | 2,048 |
| 68 | `blocks.1.11.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 69 | `blocks.1.11.mlp.2.bias` | [512] | F16 | 512 |
| 70 | `blocks.1.11.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 71 | `blocks.1.11.norm.bias` | [512] | F16 | 512 |
| 72 | `blocks.1.11.norm.weight` | [512] | F16 | 512 |
| 73 | `blocks.1.12.conv.bias` | [512] | F16 | 512 |
| 74 | `blocks.1.12.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 75 | `blocks.1.12.mlp.0.bias` | [2048] | F16 | 2,048 |
| 76 | `blocks.1.12.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 77 | `blocks.1.12.mlp.2.bias` | [512] | F16 | 512 |
| 78 | `blocks.1.12.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 79 | `blocks.1.12.norm.bias` | [512] | F16 | 512 |
| 80 | `blocks.1.12.norm.weight` | [512] | F16 | 512 |
| 81 | `blocks.1.13.conv.bias` | [512] | F16 | 512 |
| 82 | `blocks.1.13.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 83 | `blocks.1.13.mlp.0.bias` | [2048] | F16 | 2,048 |
| 84 | `blocks.1.13.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 85 | `blocks.1.13.mlp.2.bias` | [512] | F16 | 512 |
| 86 | `blocks.1.13.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 87 | `blocks.1.13.norm.bias` | [512] | F16 | 512 |
| 88 | `blocks.1.13.norm.weight` | [512] | F16 | 512 |
| 89 | `blocks.1.14.conv.bias` | [512] | F16 | 512 |
| 90 | `blocks.1.14.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 91 | `blocks.1.14.mlp.0.bias` | [2048] | F16 | 2,048 |
| 92 | `blocks.1.14.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 93 | `blocks.1.14.mlp.2.bias` | [512] | F16 | 512 |
| 94 | `blocks.1.14.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 95 | `blocks.1.14.norm.bias` | [512] | F16 | 512 |
| 96 | `blocks.1.14.norm.weight` | [512] | F16 | 512 |
| 97 | `blocks.1.15.conv.bias` | [512] | F16 | 512 |
| 98 | `blocks.1.15.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 99 | `blocks.1.15.mlp.0.bias` | [2048] | F16 | 2,048 |
| 100 | `blocks.1.15.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 101 | `blocks.1.15.mlp.2.bias` | [512] | F16 | 512 |
| 102 | `blocks.1.15.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 103 | `blocks.1.15.norm.bias` | [512] | F16 | 512 |
| 104 | `blocks.1.15.norm.weight` | [512] | F16 | 512 |
| 105 | `blocks.1.16.conv1.bias` | [2048] | F16 | 2,048 |
| 106 | `blocks.1.16.conv1.weight` | [2048, 3, 3, 3, 512] | F16 | 28,311,552 |
| 107 | `blocks.1.16.conv2.bias` | [256] | F16 | 256 |
| 108 | `blocks.1.16.conv2.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 109 | `blocks.1.16.norm1.bias` | [512] | F16 | 512 |
| 110 | `blocks.1.16.norm1.weight` | [512] | F16 | 512 |
| 111 | `blocks.1.16.to_subdiv.bias` | [8] | F16 | 8 |
| 112 | `blocks.1.16.to_subdiv.weight` | [8, 512] | F16 | 4,096 |
| 113 | `blocks.1.2.conv.bias` | [512] | F16 | 512 |
| 114 | `blocks.1.2.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 115 | `blocks.1.2.mlp.0.bias` | [2048] | F16 | 2,048 |
| 116 | `blocks.1.2.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 117 | `blocks.1.2.mlp.2.bias` | [512] | F16 | 512 |
| 118 | `blocks.1.2.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 119 | `blocks.1.2.norm.bias` | [512] | F16 | 512 |
| 120 | `blocks.1.2.norm.weight` | [512] | F16 | 512 |
| 121 | `blocks.1.3.conv.bias` | [512] | F16 | 512 |
| 122 | `blocks.1.3.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 123 | `blocks.1.3.mlp.0.bias` | [2048] | F16 | 2,048 |
| 124 | `blocks.1.3.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 125 | `blocks.1.3.mlp.2.bias` | [512] | F16 | 512 |
| 126 | `blocks.1.3.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 127 | `blocks.1.3.norm.bias` | [512] | F16 | 512 |
| 128 | `blocks.1.3.norm.weight` | [512] | F16 | 512 |
| 129 | `blocks.1.4.conv.bias` | [512] | F16 | 512 |
| 130 | `blocks.1.4.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 131 | `blocks.1.4.mlp.0.bias` | [2048] | F16 | 2,048 |
| 132 | `blocks.1.4.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 133 | `blocks.1.4.mlp.2.bias` | [512] | F16 | 512 |
| 134 | `blocks.1.4.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 135 | `blocks.1.4.norm.bias` | [512] | F16 | 512 |
| 136 | `blocks.1.4.norm.weight` | [512] | F16 | 512 |
| 137 | `blocks.1.5.conv.bias` | [512] | F16 | 512 |
| 138 | `blocks.1.5.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 139 | `blocks.1.5.mlp.0.bias` | [2048] | F16 | 2,048 |
| 140 | `blocks.1.5.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 141 | `blocks.1.5.mlp.2.bias` | [512] | F16 | 512 |
| 142 | `blocks.1.5.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 143 | `blocks.1.5.norm.bias` | [512] | F16 | 512 |
| 144 | `blocks.1.5.norm.weight` | [512] | F16 | 512 |
| 145 | `blocks.1.6.conv.bias` | [512] | F16 | 512 |
| 146 | `blocks.1.6.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 147 | `blocks.1.6.mlp.0.bias` | [2048] | F16 | 2,048 |
| 148 | `blocks.1.6.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 149 | `blocks.1.6.mlp.2.bias` | [512] | F16 | 512 |
| 150 | `blocks.1.6.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 151 | `blocks.1.6.norm.bias` | [512] | F16 | 512 |
| 152 | `blocks.1.6.norm.weight` | [512] | F16 | 512 |
| 153 | `blocks.1.7.conv.bias` | [512] | F16 | 512 |
| 154 | `blocks.1.7.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 155 | `blocks.1.7.mlp.0.bias` | [2048] | F16 | 2,048 |
| 156 | `blocks.1.7.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 157 | `blocks.1.7.mlp.2.bias` | [512] | F16 | 512 |
| 158 | `blocks.1.7.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 159 | `blocks.1.7.norm.bias` | [512] | F16 | 512 |
| 160 | `blocks.1.7.norm.weight` | [512] | F16 | 512 |
| 161 | `blocks.1.8.conv.bias` | [512] | F16 | 512 |
| 162 | `blocks.1.8.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 163 | `blocks.1.8.mlp.0.bias` | [2048] | F16 | 2,048 |
| 164 | `blocks.1.8.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 165 | `blocks.1.8.mlp.2.bias` | [512] | F16 | 512 |
| 166 | `blocks.1.8.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 167 | `blocks.1.8.norm.bias` | [512] | F16 | 512 |
| 168 | `blocks.1.8.norm.weight` | [512] | F16 | 512 |
| 169 | `blocks.1.9.conv.bias` | [512] | F16 | 512 |
| 170 | `blocks.1.9.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 171 | `blocks.1.9.mlp.0.bias` | [2048] | F16 | 2,048 |
| 172 | `blocks.1.9.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 173 | `blocks.1.9.mlp.2.bias` | [512] | F16 | 512 |
| 174 | `blocks.1.9.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 175 | `blocks.1.9.norm.bias` | [512] | F16 | 512 |
| 176 | `blocks.1.9.norm.weight` | [512] | F16 | 512 |
| 177 | `blocks.2.0.conv.bias` | [256] | F16 | 256 |
| 178 | `blocks.2.0.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 179 | `blocks.2.0.mlp.0.bias` | [1024] | F16 | 1,024 |
| 180 | `blocks.2.0.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 181 | `blocks.2.0.mlp.2.bias` | [256] | F16 | 256 |
| 182 | `blocks.2.0.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 183 | `blocks.2.0.norm.bias` | [256] | F16 | 256 |
| 184 | `blocks.2.0.norm.weight` | [256] | F16 | 256 |
| 185 | `blocks.2.1.conv.bias` | [256] | F16 | 256 |
| 186 | `blocks.2.1.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 187 | `blocks.2.1.mlp.0.bias` | [1024] | F16 | 1,024 |
| 188 | `blocks.2.1.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 189 | `blocks.2.1.mlp.2.bias` | [256] | F16 | 256 |
| 190 | `blocks.2.1.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 191 | `blocks.2.1.norm.bias` | [256] | F16 | 256 |
| 192 | `blocks.2.1.norm.weight` | [256] | F16 | 256 |
| 193 | `blocks.2.2.conv.bias` | [256] | F16 | 256 |
| 194 | `blocks.2.2.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 195 | `blocks.2.2.mlp.0.bias` | [1024] | F16 | 1,024 |
| 196 | `blocks.2.2.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 197 | `blocks.2.2.mlp.2.bias` | [256] | F16 | 256 |
| 198 | `blocks.2.2.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 199 | `blocks.2.2.norm.bias` | [256] | F16 | 256 |
| 200 | `blocks.2.2.norm.weight` | [256] | F16 | 256 |
| 201 | `blocks.2.3.conv.bias` | [256] | F16 | 256 |
| 202 | `blocks.2.3.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 203 | `blocks.2.3.mlp.0.bias` | [1024] | F16 | 1,024 |
| 204 | `blocks.2.3.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 205 | `blocks.2.3.mlp.2.bias` | [256] | F16 | 256 |
| 206 | `blocks.2.3.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 207 | `blocks.2.3.norm.bias` | [256] | F16 | 256 |
| 208 | `blocks.2.3.norm.weight` | [256] | F16 | 256 |
| 209 | `blocks.2.4.conv.bias` | [256] | F16 | 256 |
| 210 | `blocks.2.4.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 211 | `blocks.2.4.mlp.0.bias` | [1024] | F16 | 1,024 |
| 212 | `blocks.2.4.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 213 | `blocks.2.4.mlp.2.bias` | [256] | F16 | 256 |
| 214 | `blocks.2.4.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 215 | `blocks.2.4.norm.bias` | [256] | F16 | 256 |
| 216 | `blocks.2.4.norm.weight` | [256] | F16 | 256 |
| 217 | `blocks.2.5.conv.bias` | [256] | F16 | 256 |
| 218 | `blocks.2.5.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 219 | `blocks.2.5.mlp.0.bias` | [1024] | F16 | 1,024 |
| 220 | `blocks.2.5.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 221 | `blocks.2.5.mlp.2.bias` | [256] | F16 | 256 |
| 222 | `blocks.2.5.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 223 | `blocks.2.5.norm.bias` | [256] | F16 | 256 |
| 224 | `blocks.2.5.norm.weight` | [256] | F16 | 256 |
| 225 | `blocks.2.6.conv.bias` | [256] | F16 | 256 |
| 226 | `blocks.2.6.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 227 | `blocks.2.6.mlp.0.bias` | [1024] | F16 | 1,024 |
| 228 | `blocks.2.6.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 229 | `blocks.2.6.mlp.2.bias` | [256] | F16 | 256 |
| 230 | `blocks.2.6.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 231 | `blocks.2.6.norm.bias` | [256] | F16 | 256 |
| 232 | `blocks.2.6.norm.weight` | [256] | F16 | 256 |
| 233 | `blocks.2.7.conv.bias` | [256] | F16 | 256 |
| 234 | `blocks.2.7.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 235 | `blocks.2.7.mlp.0.bias` | [1024] | F16 | 1,024 |
| 236 | `blocks.2.7.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 237 | `blocks.2.7.mlp.2.bias` | [256] | F16 | 256 |
| 238 | `blocks.2.7.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 239 | `blocks.2.7.norm.bias` | [256] | F16 | 256 |
| 240 | `blocks.2.7.norm.weight` | [256] | F16 | 256 |
| 241 | `blocks.2.8.conv1.bias` | [1024] | F16 | 1,024 |
| 242 | `blocks.2.8.conv1.weight` | [1024, 3, 3, 3, 256] | F16 | 7,077,888 |
| 243 | `blocks.2.8.conv2.bias` | [128] | F16 | 128 |
| 244 | `blocks.2.8.conv2.weight` | [128, 3, 3, 3, 128] | F16 | 442,368 |
| 245 | `blocks.2.8.norm1.bias` | [256] | F16 | 256 |
| 246 | `blocks.2.8.norm1.weight` | [256] | F16 | 256 |
| 247 | `blocks.2.8.to_subdiv.bias` | [8] | F16 | 8 |
| 248 | `blocks.2.8.to_subdiv.weight` | [8, 256] | F16 | 2,048 |
| 249 | `blocks.3.0.conv.bias` | [128] | F16 | 128 |
| 250 | `blocks.3.0.conv.weight` | [128, 3, 3, 3, 128] | F16 | 442,368 |
| 251 | `blocks.3.0.mlp.0.bias` | [512] | F16 | 512 |
| 252 | `blocks.3.0.mlp.0.weight` | [512, 128] | F16 | 65,536 |
| 253 | `blocks.3.0.mlp.2.bias` | [128] | F16 | 128 |
| 254 | `blocks.3.0.mlp.2.weight` | [128, 512] | F16 | 65,536 |
| 255 | `blocks.3.0.norm.bias` | [128] | F16 | 128 |
| 256 | `blocks.3.0.norm.weight` | [128] | F16 | 128 |
| 257 | `blocks.3.1.conv.bias` | [128] | F16 | 128 |
| 258 | `blocks.3.1.conv.weight` | [128, 3, 3, 3, 128] | F16 | 442,368 |
| 259 | `blocks.3.1.mlp.0.bias` | [512] | F16 | 512 |
| 260 | `blocks.3.1.mlp.0.weight` | [512, 128] | F16 | 65,536 |
| 261 | `blocks.3.1.mlp.2.bias` | [128] | F16 | 128 |
| 262 | `blocks.3.1.mlp.2.weight` | [128, 512] | F16 | 65,536 |
| 263 | `blocks.3.1.norm.bias` | [128] | F16 | 128 |
| 264 | `blocks.3.1.norm.weight` | [128] | F16 | 128 |
| 265 | `blocks.3.2.conv.bias` | [128] | F16 | 128 |
| 266 | `blocks.3.2.conv.weight` | [128, 3, 3, 3, 128] | F16 | 442,368 |
| 267 | `blocks.3.2.mlp.0.bias` | [512] | F16 | 512 |
| 268 | `blocks.3.2.mlp.0.weight` | [512, 128] | F16 | 65,536 |
| 269 | `blocks.3.2.mlp.2.bias` | [128] | F16 | 128 |
| 270 | `blocks.3.2.mlp.2.weight` | [128, 512] | F16 | 65,536 |
| 271 | `blocks.3.2.norm.bias` | [128] | F16 | 128 |
| 272 | `blocks.3.2.norm.weight` | [128] | F16 | 128 |
| 273 | `blocks.3.3.conv.bias` | [128] | F16 | 128 |
| 274 | `blocks.3.3.conv.weight` | [128, 3, 3, 3, 128] | F16 | 442,368 |
| 275 | `blocks.3.3.mlp.0.bias` | [512] | F16 | 512 |
| 276 | `blocks.3.3.mlp.0.weight` | [512, 128] | F16 | 65,536 |
| 277 | `blocks.3.3.mlp.2.bias` | [128] | F16 | 128 |
| 278 | `blocks.3.3.mlp.2.weight` | [128, 512] | F16 | 65,536 |
| 279 | `blocks.3.3.norm.bias` | [128] | F16 | 128 |
| 280 | `blocks.3.3.norm.weight` | [128] | F16 | 128 |
| 281 | `blocks.3.4.conv1.bias` | [512] | F16 | 512 |
| 282 | `blocks.3.4.conv1.weight` | [512, 3, 3, 3, 128] | F16 | 1,769,472 |
| 283 | `blocks.3.4.conv2.bias` | [64] | F16 | 64 |
| 284 | `blocks.3.4.conv2.weight` | [64, 3, 3, 3, 64] | F16 | 110,592 |
| 285 | `blocks.3.4.norm1.bias` | [128] | F16 | 128 |
| 286 | `blocks.3.4.norm1.weight` | [128] | F16 | 128 |
| 287 | `blocks.3.4.to_subdiv.bias` | [8] | F16 | 8 |
| 288 | `blocks.3.4.to_subdiv.weight` | [8, 128] | F16 | 1,024 |
| 289 | `from_latent.bias` | [1024] | F16 | 1,024 |
| 290 | `from_latent.weight` | [1024, 32] | F16 | 32,768 |
| 291 | `output_layer.bias` | [7] | F16 | 7 |
| 292 | `output_layer.weight` | [7, 64] | F16 | 448 |

## `tex_enc_next_dc_f16c32_fp16.safetensors`

**Role:** SC-VAE material encoder

SparseUnetVaeEncoder: same backbone as shape encoder; in_channels=6 (PBR: base_color 3 + metallic 1 + roughness 1 + alpha 1). Training-only.

Total: **284 parameters**, 354.39M elements.

Top-level prefixes: `blocks` (280), `input_layer` (2), `to_latent` (2)

| # | Parameter | Shape | Dtype | Elements |
|---:|---|---|:---:|---:|
| 1 | `blocks.0.0.conv1.bias` | [16] | F16 | 16 |
| 2 | `blocks.0.0.conv1.weight` | [16, 3, 3, 3, 64] | F16 | 27,648 |
| 3 | `blocks.0.0.conv2.bias` | [128] | F16 | 128 |
| 4 | `blocks.0.0.conv2.weight` | [128, 3, 3, 3, 128] | F16 | 442,368 |
| 5 | `blocks.0.0.norm1.bias` | [64] | F16 | 64 |
| 6 | `blocks.0.0.norm1.weight` | [64] | F16 | 64 |
| 7 | `blocks.1.0.conv.bias` | [128] | F16 | 128 |
| 8 | `blocks.1.0.conv.weight` | [128, 3, 3, 3, 128] | F16 | 442,368 |
| 9 | `blocks.1.0.mlp.0.bias` | [512] | F16 | 512 |
| 10 | `blocks.1.0.mlp.0.weight` | [512, 128] | F16 | 65,536 |
| 11 | `blocks.1.0.mlp.2.bias` | [128] | F16 | 128 |
| 12 | `blocks.1.0.mlp.2.weight` | [128, 512] | F16 | 65,536 |
| 13 | `blocks.1.0.norm.bias` | [128] | F16 | 128 |
| 14 | `blocks.1.0.norm.weight` | [128] | F16 | 128 |
| 15 | `blocks.1.1.conv.bias` | [128] | F16 | 128 |
| 16 | `blocks.1.1.conv.weight` | [128, 3, 3, 3, 128] | F16 | 442,368 |
| 17 | `blocks.1.1.mlp.0.bias` | [512] | F16 | 512 |
| 18 | `blocks.1.1.mlp.0.weight` | [512, 128] | F16 | 65,536 |
| 19 | `blocks.1.1.mlp.2.bias` | [128] | F16 | 128 |
| 20 | `blocks.1.1.mlp.2.weight` | [128, 512] | F16 | 65,536 |
| 21 | `blocks.1.1.norm.bias` | [128] | F16 | 128 |
| 22 | `blocks.1.1.norm.weight` | [128] | F16 | 128 |
| 23 | `blocks.1.2.conv.bias` | [128] | F16 | 128 |
| 24 | `blocks.1.2.conv.weight` | [128, 3, 3, 3, 128] | F16 | 442,368 |
| 25 | `blocks.1.2.mlp.0.bias` | [512] | F16 | 512 |
| 26 | `blocks.1.2.mlp.0.weight` | [512, 128] | F16 | 65,536 |
| 27 | `blocks.1.2.mlp.2.bias` | [128] | F16 | 128 |
| 28 | `blocks.1.2.mlp.2.weight` | [128, 512] | F16 | 65,536 |
| 29 | `blocks.1.2.norm.bias` | [128] | F16 | 128 |
| 30 | `blocks.1.2.norm.weight` | [128] | F16 | 128 |
| 31 | `blocks.1.3.conv.bias` | [128] | F16 | 128 |
| 32 | `blocks.1.3.conv.weight` | [128, 3, 3, 3, 128] | F16 | 442,368 |
| 33 | `blocks.1.3.mlp.0.bias` | [512] | F16 | 512 |
| 34 | `blocks.1.3.mlp.0.weight` | [512, 128] | F16 | 65,536 |
| 35 | `blocks.1.3.mlp.2.bias` | [128] | F16 | 128 |
| 36 | `blocks.1.3.mlp.2.weight` | [128, 512] | F16 | 65,536 |
| 37 | `blocks.1.3.norm.bias` | [128] | F16 | 128 |
| 38 | `blocks.1.3.norm.weight` | [128] | F16 | 128 |
| 39 | `blocks.1.4.conv1.bias` | [32] | F16 | 32 |
| 40 | `blocks.1.4.conv1.weight` | [32, 3, 3, 3, 128] | F16 | 110,592 |
| 41 | `blocks.1.4.conv2.bias` | [256] | F16 | 256 |
| 42 | `blocks.1.4.conv2.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 43 | `blocks.1.4.norm1.bias` | [128] | F16 | 128 |
| 44 | `blocks.1.4.norm1.weight` | [128] | F16 | 128 |
| 45 | `blocks.2.0.conv.bias` | [256] | F16 | 256 |
| 46 | `blocks.2.0.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 47 | `blocks.2.0.mlp.0.bias` | [1024] | F16 | 1,024 |
| 48 | `blocks.2.0.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 49 | `blocks.2.0.mlp.2.bias` | [256] | F16 | 256 |
| 50 | `blocks.2.0.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 51 | `blocks.2.0.norm.bias` | [256] | F16 | 256 |
| 52 | `blocks.2.0.norm.weight` | [256] | F16 | 256 |
| 53 | `blocks.2.1.conv.bias` | [256] | F16 | 256 |
| 54 | `blocks.2.1.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 55 | `blocks.2.1.mlp.0.bias` | [1024] | F16 | 1,024 |
| 56 | `blocks.2.1.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 57 | `blocks.2.1.mlp.2.bias` | [256] | F16 | 256 |
| 58 | `blocks.2.1.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 59 | `blocks.2.1.norm.bias` | [256] | F16 | 256 |
| 60 | `blocks.2.1.norm.weight` | [256] | F16 | 256 |
| 61 | `blocks.2.2.conv.bias` | [256] | F16 | 256 |
| 62 | `blocks.2.2.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 63 | `blocks.2.2.mlp.0.bias` | [1024] | F16 | 1,024 |
| 64 | `blocks.2.2.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 65 | `blocks.2.2.mlp.2.bias` | [256] | F16 | 256 |
| 66 | `blocks.2.2.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 67 | `blocks.2.2.norm.bias` | [256] | F16 | 256 |
| 68 | `blocks.2.2.norm.weight` | [256] | F16 | 256 |
| 69 | `blocks.2.3.conv.bias` | [256] | F16 | 256 |
| 70 | `blocks.2.3.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 71 | `blocks.2.3.mlp.0.bias` | [1024] | F16 | 1,024 |
| 72 | `blocks.2.3.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 73 | `blocks.2.3.mlp.2.bias` | [256] | F16 | 256 |
| 74 | `blocks.2.3.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 75 | `blocks.2.3.norm.bias` | [256] | F16 | 256 |
| 76 | `blocks.2.3.norm.weight` | [256] | F16 | 256 |
| 77 | `blocks.2.4.conv.bias` | [256] | F16 | 256 |
| 78 | `blocks.2.4.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 79 | `blocks.2.4.mlp.0.bias` | [1024] | F16 | 1,024 |
| 80 | `blocks.2.4.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 81 | `blocks.2.4.mlp.2.bias` | [256] | F16 | 256 |
| 82 | `blocks.2.4.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 83 | `blocks.2.4.norm.bias` | [256] | F16 | 256 |
| 84 | `blocks.2.4.norm.weight` | [256] | F16 | 256 |
| 85 | `blocks.2.5.conv.bias` | [256] | F16 | 256 |
| 86 | `blocks.2.5.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 87 | `blocks.2.5.mlp.0.bias` | [1024] | F16 | 1,024 |
| 88 | `blocks.2.5.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 89 | `blocks.2.5.mlp.2.bias` | [256] | F16 | 256 |
| 90 | `blocks.2.5.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 91 | `blocks.2.5.norm.bias` | [256] | F16 | 256 |
| 92 | `blocks.2.5.norm.weight` | [256] | F16 | 256 |
| 93 | `blocks.2.6.conv.bias` | [256] | F16 | 256 |
| 94 | `blocks.2.6.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 95 | `blocks.2.6.mlp.0.bias` | [1024] | F16 | 1,024 |
| 96 | `blocks.2.6.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 97 | `blocks.2.6.mlp.2.bias` | [256] | F16 | 256 |
| 98 | `blocks.2.6.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 99 | `blocks.2.6.norm.bias` | [256] | F16 | 256 |
| 100 | `blocks.2.6.norm.weight` | [256] | F16 | 256 |
| 101 | `blocks.2.7.conv.bias` | [256] | F16 | 256 |
| 102 | `blocks.2.7.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 103 | `blocks.2.7.mlp.0.bias` | [1024] | F16 | 1,024 |
| 104 | `blocks.2.7.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 105 | `blocks.2.7.mlp.2.bias` | [256] | F16 | 256 |
| 106 | `blocks.2.7.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 107 | `blocks.2.7.norm.bias` | [256] | F16 | 256 |
| 108 | `blocks.2.7.norm.weight` | [256] | F16 | 256 |
| 109 | `blocks.2.8.conv1.bias` | [64] | F16 | 64 |
| 110 | `blocks.2.8.conv1.weight` | [64, 3, 3, 3, 256] | F16 | 442,368 |
| 111 | `blocks.2.8.conv2.bias` | [512] | F16 | 512 |
| 112 | `blocks.2.8.conv2.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 113 | `blocks.2.8.norm1.bias` | [256] | F16 | 256 |
| 114 | `blocks.2.8.norm1.weight` | [256] | F16 | 256 |
| 115 | `blocks.3.0.conv.bias` | [512] | F16 | 512 |
| 116 | `blocks.3.0.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 117 | `blocks.3.0.mlp.0.bias` | [2048] | F16 | 2,048 |
| 118 | `blocks.3.0.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 119 | `blocks.3.0.mlp.2.bias` | [512] | F16 | 512 |
| 120 | `blocks.3.0.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 121 | `blocks.3.0.norm.bias` | [512] | F16 | 512 |
| 122 | `blocks.3.0.norm.weight` | [512] | F16 | 512 |
| 123 | `blocks.3.1.conv.bias` | [512] | F16 | 512 |
| 124 | `blocks.3.1.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 125 | `blocks.3.1.mlp.0.bias` | [2048] | F16 | 2,048 |
| 126 | `blocks.3.1.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 127 | `blocks.3.1.mlp.2.bias` | [512] | F16 | 512 |
| 128 | `blocks.3.1.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 129 | `blocks.3.1.norm.bias` | [512] | F16 | 512 |
| 130 | `blocks.3.1.norm.weight` | [512] | F16 | 512 |
| 131 | `blocks.3.10.conv.bias` | [512] | F16 | 512 |
| 132 | `blocks.3.10.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 133 | `blocks.3.10.mlp.0.bias` | [2048] | F16 | 2,048 |
| 134 | `blocks.3.10.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 135 | `blocks.3.10.mlp.2.bias` | [512] | F16 | 512 |
| 136 | `blocks.3.10.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 137 | `blocks.3.10.norm.bias` | [512] | F16 | 512 |
| 138 | `blocks.3.10.norm.weight` | [512] | F16 | 512 |
| 139 | `blocks.3.11.conv.bias` | [512] | F16 | 512 |
| 140 | `blocks.3.11.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 141 | `blocks.3.11.mlp.0.bias` | [2048] | F16 | 2,048 |
| 142 | `blocks.3.11.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 143 | `blocks.3.11.mlp.2.bias` | [512] | F16 | 512 |
| 144 | `blocks.3.11.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 145 | `blocks.3.11.norm.bias` | [512] | F16 | 512 |
| 146 | `blocks.3.11.norm.weight` | [512] | F16 | 512 |
| 147 | `blocks.3.12.conv.bias` | [512] | F16 | 512 |
| 148 | `blocks.3.12.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 149 | `blocks.3.12.mlp.0.bias` | [2048] | F16 | 2,048 |
| 150 | `blocks.3.12.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 151 | `blocks.3.12.mlp.2.bias` | [512] | F16 | 512 |
| 152 | `blocks.3.12.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 153 | `blocks.3.12.norm.bias` | [512] | F16 | 512 |
| 154 | `blocks.3.12.norm.weight` | [512] | F16 | 512 |
| 155 | `blocks.3.13.conv.bias` | [512] | F16 | 512 |
| 156 | `blocks.3.13.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 157 | `blocks.3.13.mlp.0.bias` | [2048] | F16 | 2,048 |
| 158 | `blocks.3.13.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 159 | `blocks.3.13.mlp.2.bias` | [512] | F16 | 512 |
| 160 | `blocks.3.13.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 161 | `blocks.3.13.norm.bias` | [512] | F16 | 512 |
| 162 | `blocks.3.13.norm.weight` | [512] | F16 | 512 |
| 163 | `blocks.3.14.conv.bias` | [512] | F16 | 512 |
| 164 | `blocks.3.14.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 165 | `blocks.3.14.mlp.0.bias` | [2048] | F16 | 2,048 |
| 166 | `blocks.3.14.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 167 | `blocks.3.14.mlp.2.bias` | [512] | F16 | 512 |
| 168 | `blocks.3.14.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 169 | `blocks.3.14.norm.bias` | [512] | F16 | 512 |
| 170 | `blocks.3.14.norm.weight` | [512] | F16 | 512 |
| 171 | `blocks.3.15.conv.bias` | [512] | F16 | 512 |
| 172 | `blocks.3.15.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 173 | `blocks.3.15.mlp.0.bias` | [2048] | F16 | 2,048 |
| 174 | `blocks.3.15.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 175 | `blocks.3.15.mlp.2.bias` | [512] | F16 | 512 |
| 176 | `blocks.3.15.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 177 | `blocks.3.15.norm.bias` | [512] | F16 | 512 |
| 178 | `blocks.3.15.norm.weight` | [512] | F16 | 512 |
| 179 | `blocks.3.16.conv1.bias` | [128] | F16 | 128 |
| 180 | `blocks.3.16.conv1.weight` | [128, 3, 3, 3, 512] | F16 | 1,769,472 |
| 181 | `blocks.3.16.conv2.bias` | [1024] | F16 | 1,024 |
| 182 | `blocks.3.16.conv2.weight` | [1024, 3, 3, 3, 1024] | F16 | 28,311,552 |
| 183 | `blocks.3.16.norm1.bias` | [512] | F16 | 512 |
| 184 | `blocks.3.16.norm1.weight` | [512] | F16 | 512 |
| 185 | `blocks.3.2.conv.bias` | [512] | F16 | 512 |
| 186 | `blocks.3.2.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 187 | `blocks.3.2.mlp.0.bias` | [2048] | F16 | 2,048 |
| 188 | `blocks.3.2.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 189 | `blocks.3.2.mlp.2.bias` | [512] | F16 | 512 |
| 190 | `blocks.3.2.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 191 | `blocks.3.2.norm.bias` | [512] | F16 | 512 |
| 192 | `blocks.3.2.norm.weight` | [512] | F16 | 512 |
| 193 | `blocks.3.3.conv.bias` | [512] | F16 | 512 |
| 194 | `blocks.3.3.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 195 | `blocks.3.3.mlp.0.bias` | [2048] | F16 | 2,048 |
| 196 | `blocks.3.3.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 197 | `blocks.3.3.mlp.2.bias` | [512] | F16 | 512 |
| 198 | `blocks.3.3.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 199 | `blocks.3.3.norm.bias` | [512] | F16 | 512 |
| 200 | `blocks.3.3.norm.weight` | [512] | F16 | 512 |
| 201 | `blocks.3.4.conv.bias` | [512] | F16 | 512 |
| 202 | `blocks.3.4.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 203 | `blocks.3.4.mlp.0.bias` | [2048] | F16 | 2,048 |
| 204 | `blocks.3.4.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 205 | `blocks.3.4.mlp.2.bias` | [512] | F16 | 512 |
| 206 | `blocks.3.4.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 207 | `blocks.3.4.norm.bias` | [512] | F16 | 512 |
| 208 | `blocks.3.4.norm.weight` | [512] | F16 | 512 |
| 209 | `blocks.3.5.conv.bias` | [512] | F16 | 512 |
| 210 | `blocks.3.5.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 211 | `blocks.3.5.mlp.0.bias` | [2048] | F16 | 2,048 |
| 212 | `blocks.3.5.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 213 | `blocks.3.5.mlp.2.bias` | [512] | F16 | 512 |
| 214 | `blocks.3.5.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 215 | `blocks.3.5.norm.bias` | [512] | F16 | 512 |
| 216 | `blocks.3.5.norm.weight` | [512] | F16 | 512 |
| 217 | `blocks.3.6.conv.bias` | [512] | F16 | 512 |
| 218 | `blocks.3.6.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 219 | `blocks.3.6.mlp.0.bias` | [2048] | F16 | 2,048 |
| 220 | `blocks.3.6.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 221 | `blocks.3.6.mlp.2.bias` | [512] | F16 | 512 |
| 222 | `blocks.3.6.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 223 | `blocks.3.6.norm.bias` | [512] | F16 | 512 |
| 224 | `blocks.3.6.norm.weight` | [512] | F16 | 512 |
| 225 | `blocks.3.7.conv.bias` | [512] | F16 | 512 |
| 226 | `blocks.3.7.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 227 | `blocks.3.7.mlp.0.bias` | [2048] | F16 | 2,048 |
| 228 | `blocks.3.7.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 229 | `blocks.3.7.mlp.2.bias` | [512] | F16 | 512 |
| 230 | `blocks.3.7.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 231 | `blocks.3.7.norm.bias` | [512] | F16 | 512 |
| 232 | `blocks.3.7.norm.weight` | [512] | F16 | 512 |
| 233 | `blocks.3.8.conv.bias` | [512] | F16 | 512 |
| 234 | `blocks.3.8.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 235 | `blocks.3.8.mlp.0.bias` | [2048] | F16 | 2,048 |
| 236 | `blocks.3.8.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 237 | `blocks.3.8.mlp.2.bias` | [512] | F16 | 512 |
| 238 | `blocks.3.8.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 239 | `blocks.3.8.norm.bias` | [512] | F16 | 512 |
| 240 | `blocks.3.8.norm.weight` | [512] | F16 | 512 |
| 241 | `blocks.3.9.conv.bias` | [512] | F16 | 512 |
| 242 | `blocks.3.9.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 243 | `blocks.3.9.mlp.0.bias` | [2048] | F16 | 2,048 |
| 244 | `blocks.3.9.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 245 | `blocks.3.9.mlp.2.bias` | [512] | F16 | 512 |
| 246 | `blocks.3.9.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 247 | `blocks.3.9.norm.bias` | [512] | F16 | 512 |
| 248 | `blocks.3.9.norm.weight` | [512] | F16 | 512 |
| 249 | `blocks.4.0.conv.bias` | [1024] | F16 | 1,024 |
| 250 | `blocks.4.0.conv.weight` | [1024, 3, 3, 3, 1024] | F16 | 28,311,552 |
| 251 | `blocks.4.0.mlp.0.bias` | [4096] | F16 | 4,096 |
| 252 | `blocks.4.0.mlp.0.weight` | [4096, 1024] | F16 | 4,194,304 |
| 253 | `blocks.4.0.mlp.2.bias` | [1024] | F16 | 1,024 |
| 254 | `blocks.4.0.mlp.2.weight` | [1024, 4096] | F16 | 4,194,304 |
| 255 | `blocks.4.0.norm.bias` | [1024] | F16 | 1,024 |
| 256 | `blocks.4.0.norm.weight` | [1024] | F16 | 1,024 |
| 257 | `blocks.4.1.conv.bias` | [1024] | F16 | 1,024 |
| 258 | `blocks.4.1.conv.weight` | [1024, 3, 3, 3, 1024] | F16 | 28,311,552 |
| 259 | `blocks.4.1.mlp.0.bias` | [4096] | F16 | 4,096 |
| 260 | `blocks.4.1.mlp.0.weight` | [4096, 1024] | F16 | 4,194,304 |
| 261 | `blocks.4.1.mlp.2.bias` | [1024] | F16 | 1,024 |
| 262 | `blocks.4.1.mlp.2.weight` | [1024, 4096] | F16 | 4,194,304 |
| 263 | `blocks.4.1.norm.bias` | [1024] | F16 | 1,024 |
| 264 | `blocks.4.1.norm.weight` | [1024] | F16 | 1,024 |
| 265 | `blocks.4.2.conv.bias` | [1024] | F16 | 1,024 |
| 266 | `blocks.4.2.conv.weight` | [1024, 3, 3, 3, 1024] | F16 | 28,311,552 |
| 267 | `blocks.4.2.mlp.0.bias` | [4096] | F16 | 4,096 |
| 268 | `blocks.4.2.mlp.0.weight` | [4096, 1024] | F16 | 4,194,304 |
| 269 | `blocks.4.2.mlp.2.bias` | [1024] | F16 | 1,024 |
| 270 | `blocks.4.2.mlp.2.weight` | [1024, 4096] | F16 | 4,194,304 |
| 271 | `blocks.4.2.norm.bias` | [1024] | F16 | 1,024 |
| 272 | `blocks.4.2.norm.weight` | [1024] | F16 | 1,024 |
| 273 | `blocks.4.3.conv.bias` | [1024] | F16 | 1,024 |
| 274 | `blocks.4.3.conv.weight` | [1024, 3, 3, 3, 1024] | F16 | 28,311,552 |
| 275 | `blocks.4.3.mlp.0.bias` | [4096] | F16 | 4,096 |
| 276 | `blocks.4.3.mlp.0.weight` | [4096, 1024] | F16 | 4,194,304 |
| 277 | `blocks.4.3.mlp.2.bias` | [1024] | F16 | 1,024 |
| 278 | `blocks.4.3.mlp.2.weight` | [1024, 4096] | F16 | 4,194,304 |
| 279 | `blocks.4.3.norm.bias` | [1024] | F16 | 1,024 |
| 280 | `blocks.4.3.norm.weight` | [1024] | F16 | 1,024 |
| 281 | `input_layer.bias` | [64] | F16 | 64 |
| 282 | `input_layer.weight` | [64, 6] | F16 | 384 |
| 283 | `to_latent.bias` | [64] | F16 | 64 |
| 284 | `to_latent.weight` | [64, 1024] | F16 | 65,536 |

## `tex_dec_next_dc_f16c32_fp16.safetensors`

**Role:** SC-VAE material decoder

SparseUnetVaeDecoder: produces per-active-voxel (c, m, r, α). Conditioned on the shape decoder's sub-structure (guide_subs) so the active set matches the geometry.

Total: **284 parameters**, 474.22M elements.

Top-level prefixes: `blocks` (280), `from_latent` (2), `output_layer` (2)

| # | Parameter | Shape | Dtype | Elements |
|---:|---|---|:---:|---:|
| 1 | `blocks.0.0.conv.bias` | [1024] | F16 | 1,024 |
| 2 | `blocks.0.0.conv.weight` | [1024, 3, 3, 3, 1024] | F16 | 28,311,552 |
| 3 | `blocks.0.0.mlp.0.bias` | [4096] | F16 | 4,096 |
| 4 | `blocks.0.0.mlp.0.weight` | [4096, 1024] | F16 | 4,194,304 |
| 5 | `blocks.0.0.mlp.2.bias` | [1024] | F16 | 1,024 |
| 6 | `blocks.0.0.mlp.2.weight` | [1024, 4096] | F16 | 4,194,304 |
| 7 | `blocks.0.0.norm.bias` | [1024] | F16 | 1,024 |
| 8 | `blocks.0.0.norm.weight` | [1024] | F16 | 1,024 |
| 9 | `blocks.0.1.conv.bias` | [1024] | F16 | 1,024 |
| 10 | `blocks.0.1.conv.weight` | [1024, 3, 3, 3, 1024] | F16 | 28,311,552 |
| 11 | `blocks.0.1.mlp.0.bias` | [4096] | F16 | 4,096 |
| 12 | `blocks.0.1.mlp.0.weight` | [4096, 1024] | F16 | 4,194,304 |
| 13 | `blocks.0.1.mlp.2.bias` | [1024] | F16 | 1,024 |
| 14 | `blocks.0.1.mlp.2.weight` | [1024, 4096] | F16 | 4,194,304 |
| 15 | `blocks.0.1.norm.bias` | [1024] | F16 | 1,024 |
| 16 | `blocks.0.1.norm.weight` | [1024] | F16 | 1,024 |
| 17 | `blocks.0.2.conv.bias` | [1024] | F16 | 1,024 |
| 18 | `blocks.0.2.conv.weight` | [1024, 3, 3, 3, 1024] | F16 | 28,311,552 |
| 19 | `blocks.0.2.mlp.0.bias` | [4096] | F16 | 4,096 |
| 20 | `blocks.0.2.mlp.0.weight` | [4096, 1024] | F16 | 4,194,304 |
| 21 | `blocks.0.2.mlp.2.bias` | [1024] | F16 | 1,024 |
| 22 | `blocks.0.2.mlp.2.weight` | [1024, 4096] | F16 | 4,194,304 |
| 23 | `blocks.0.2.norm.bias` | [1024] | F16 | 1,024 |
| 24 | `blocks.0.2.norm.weight` | [1024] | F16 | 1,024 |
| 25 | `blocks.0.3.conv.bias` | [1024] | F16 | 1,024 |
| 26 | `blocks.0.3.conv.weight` | [1024, 3, 3, 3, 1024] | F16 | 28,311,552 |
| 27 | `blocks.0.3.mlp.0.bias` | [4096] | F16 | 4,096 |
| 28 | `blocks.0.3.mlp.0.weight` | [4096, 1024] | F16 | 4,194,304 |
| 29 | `blocks.0.3.mlp.2.bias` | [1024] | F16 | 1,024 |
| 30 | `blocks.0.3.mlp.2.weight` | [1024, 4096] | F16 | 4,194,304 |
| 31 | `blocks.0.3.norm.bias` | [1024] | F16 | 1,024 |
| 32 | `blocks.0.3.norm.weight` | [1024] | F16 | 1,024 |
| 33 | `blocks.0.4.conv1.bias` | [4096] | F16 | 4,096 |
| 34 | `blocks.0.4.conv1.weight` | [4096, 3, 3, 3, 1024] | F16 | 113,246,208 |
| 35 | `blocks.0.4.conv2.bias` | [512] | F16 | 512 |
| 36 | `blocks.0.4.conv2.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 37 | `blocks.0.4.norm1.bias` | [1024] | F16 | 1,024 |
| 38 | `blocks.0.4.norm1.weight` | [1024] | F16 | 1,024 |
| 39 | `blocks.1.0.conv.bias` | [512] | F16 | 512 |
| 40 | `blocks.1.0.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 41 | `blocks.1.0.mlp.0.bias` | [2048] | F16 | 2,048 |
| 42 | `blocks.1.0.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 43 | `blocks.1.0.mlp.2.bias` | [512] | F16 | 512 |
| 44 | `blocks.1.0.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 45 | `blocks.1.0.norm.bias` | [512] | F16 | 512 |
| 46 | `blocks.1.0.norm.weight` | [512] | F16 | 512 |
| 47 | `blocks.1.1.conv.bias` | [512] | F16 | 512 |
| 48 | `blocks.1.1.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 49 | `blocks.1.1.mlp.0.bias` | [2048] | F16 | 2,048 |
| 50 | `blocks.1.1.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 51 | `blocks.1.1.mlp.2.bias` | [512] | F16 | 512 |
| 52 | `blocks.1.1.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 53 | `blocks.1.1.norm.bias` | [512] | F16 | 512 |
| 54 | `blocks.1.1.norm.weight` | [512] | F16 | 512 |
| 55 | `blocks.1.10.conv.bias` | [512] | F16 | 512 |
| 56 | `blocks.1.10.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 57 | `blocks.1.10.mlp.0.bias` | [2048] | F16 | 2,048 |
| 58 | `blocks.1.10.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 59 | `blocks.1.10.mlp.2.bias` | [512] | F16 | 512 |
| 60 | `blocks.1.10.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 61 | `blocks.1.10.norm.bias` | [512] | F16 | 512 |
| 62 | `blocks.1.10.norm.weight` | [512] | F16 | 512 |
| 63 | `blocks.1.11.conv.bias` | [512] | F16 | 512 |
| 64 | `blocks.1.11.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 65 | `blocks.1.11.mlp.0.bias` | [2048] | F16 | 2,048 |
| 66 | `blocks.1.11.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 67 | `blocks.1.11.mlp.2.bias` | [512] | F16 | 512 |
| 68 | `blocks.1.11.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 69 | `blocks.1.11.norm.bias` | [512] | F16 | 512 |
| 70 | `blocks.1.11.norm.weight` | [512] | F16 | 512 |
| 71 | `blocks.1.12.conv.bias` | [512] | F16 | 512 |
| 72 | `blocks.1.12.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 73 | `blocks.1.12.mlp.0.bias` | [2048] | F16 | 2,048 |
| 74 | `blocks.1.12.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 75 | `blocks.1.12.mlp.2.bias` | [512] | F16 | 512 |
| 76 | `blocks.1.12.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 77 | `blocks.1.12.norm.bias` | [512] | F16 | 512 |
| 78 | `blocks.1.12.norm.weight` | [512] | F16 | 512 |
| 79 | `blocks.1.13.conv.bias` | [512] | F16 | 512 |
| 80 | `blocks.1.13.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 81 | `blocks.1.13.mlp.0.bias` | [2048] | F16 | 2,048 |
| 82 | `blocks.1.13.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 83 | `blocks.1.13.mlp.2.bias` | [512] | F16 | 512 |
| 84 | `blocks.1.13.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 85 | `blocks.1.13.norm.bias` | [512] | F16 | 512 |
| 86 | `blocks.1.13.norm.weight` | [512] | F16 | 512 |
| 87 | `blocks.1.14.conv.bias` | [512] | F16 | 512 |
| 88 | `blocks.1.14.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 89 | `blocks.1.14.mlp.0.bias` | [2048] | F16 | 2,048 |
| 90 | `blocks.1.14.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 91 | `blocks.1.14.mlp.2.bias` | [512] | F16 | 512 |
| 92 | `blocks.1.14.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 93 | `blocks.1.14.norm.bias` | [512] | F16 | 512 |
| 94 | `blocks.1.14.norm.weight` | [512] | F16 | 512 |
| 95 | `blocks.1.15.conv.bias` | [512] | F16 | 512 |
| 96 | `blocks.1.15.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 97 | `blocks.1.15.mlp.0.bias` | [2048] | F16 | 2,048 |
| 98 | `blocks.1.15.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 99 | `blocks.1.15.mlp.2.bias` | [512] | F16 | 512 |
| 100 | `blocks.1.15.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 101 | `blocks.1.15.norm.bias` | [512] | F16 | 512 |
| 102 | `blocks.1.15.norm.weight` | [512] | F16 | 512 |
| 103 | `blocks.1.16.conv1.bias` | [2048] | F16 | 2,048 |
| 104 | `blocks.1.16.conv1.weight` | [2048, 3, 3, 3, 512] | F16 | 28,311,552 |
| 105 | `blocks.1.16.conv2.bias` | [256] | F16 | 256 |
| 106 | `blocks.1.16.conv2.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 107 | `blocks.1.16.norm1.bias` | [512] | F16 | 512 |
| 108 | `blocks.1.16.norm1.weight` | [512] | F16 | 512 |
| 109 | `blocks.1.2.conv.bias` | [512] | F16 | 512 |
| 110 | `blocks.1.2.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 111 | `blocks.1.2.mlp.0.bias` | [2048] | F16 | 2,048 |
| 112 | `blocks.1.2.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 113 | `blocks.1.2.mlp.2.bias` | [512] | F16 | 512 |
| 114 | `blocks.1.2.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 115 | `blocks.1.2.norm.bias` | [512] | F16 | 512 |
| 116 | `blocks.1.2.norm.weight` | [512] | F16 | 512 |
| 117 | `blocks.1.3.conv.bias` | [512] | F16 | 512 |
| 118 | `blocks.1.3.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 119 | `blocks.1.3.mlp.0.bias` | [2048] | F16 | 2,048 |
| 120 | `blocks.1.3.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 121 | `blocks.1.3.mlp.2.bias` | [512] | F16 | 512 |
| 122 | `blocks.1.3.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 123 | `blocks.1.3.norm.bias` | [512] | F16 | 512 |
| 124 | `blocks.1.3.norm.weight` | [512] | F16 | 512 |
| 125 | `blocks.1.4.conv.bias` | [512] | F16 | 512 |
| 126 | `blocks.1.4.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 127 | `blocks.1.4.mlp.0.bias` | [2048] | F16 | 2,048 |
| 128 | `blocks.1.4.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 129 | `blocks.1.4.mlp.2.bias` | [512] | F16 | 512 |
| 130 | `blocks.1.4.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 131 | `blocks.1.4.norm.bias` | [512] | F16 | 512 |
| 132 | `blocks.1.4.norm.weight` | [512] | F16 | 512 |
| 133 | `blocks.1.5.conv.bias` | [512] | F16 | 512 |
| 134 | `blocks.1.5.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 135 | `blocks.1.5.mlp.0.bias` | [2048] | F16 | 2,048 |
| 136 | `blocks.1.5.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 137 | `blocks.1.5.mlp.2.bias` | [512] | F16 | 512 |
| 138 | `blocks.1.5.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 139 | `blocks.1.5.norm.bias` | [512] | F16 | 512 |
| 140 | `blocks.1.5.norm.weight` | [512] | F16 | 512 |
| 141 | `blocks.1.6.conv.bias` | [512] | F16 | 512 |
| 142 | `blocks.1.6.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 143 | `blocks.1.6.mlp.0.bias` | [2048] | F16 | 2,048 |
| 144 | `blocks.1.6.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 145 | `blocks.1.6.mlp.2.bias` | [512] | F16 | 512 |
| 146 | `blocks.1.6.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 147 | `blocks.1.6.norm.bias` | [512] | F16 | 512 |
| 148 | `blocks.1.6.norm.weight` | [512] | F16 | 512 |
| 149 | `blocks.1.7.conv.bias` | [512] | F16 | 512 |
| 150 | `blocks.1.7.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 151 | `blocks.1.7.mlp.0.bias` | [2048] | F16 | 2,048 |
| 152 | `blocks.1.7.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 153 | `blocks.1.7.mlp.2.bias` | [512] | F16 | 512 |
| 154 | `blocks.1.7.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 155 | `blocks.1.7.norm.bias` | [512] | F16 | 512 |
| 156 | `blocks.1.7.norm.weight` | [512] | F16 | 512 |
| 157 | `blocks.1.8.conv.bias` | [512] | F16 | 512 |
| 158 | `blocks.1.8.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 159 | `blocks.1.8.mlp.0.bias` | [2048] | F16 | 2,048 |
| 160 | `blocks.1.8.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 161 | `blocks.1.8.mlp.2.bias` | [512] | F16 | 512 |
| 162 | `blocks.1.8.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 163 | `blocks.1.8.norm.bias` | [512] | F16 | 512 |
| 164 | `blocks.1.8.norm.weight` | [512] | F16 | 512 |
| 165 | `blocks.1.9.conv.bias` | [512] | F16 | 512 |
| 166 | `blocks.1.9.conv.weight` | [512, 3, 3, 3, 512] | F16 | 7,077,888 |
| 167 | `blocks.1.9.mlp.0.bias` | [2048] | F16 | 2,048 |
| 168 | `blocks.1.9.mlp.0.weight` | [2048, 512] | F16 | 1,048,576 |
| 169 | `blocks.1.9.mlp.2.bias` | [512] | F16 | 512 |
| 170 | `blocks.1.9.mlp.2.weight` | [512, 2048] | F16 | 1,048,576 |
| 171 | `blocks.1.9.norm.bias` | [512] | F16 | 512 |
| 172 | `blocks.1.9.norm.weight` | [512] | F16 | 512 |
| 173 | `blocks.2.0.conv.bias` | [256] | F16 | 256 |
| 174 | `blocks.2.0.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 175 | `blocks.2.0.mlp.0.bias` | [1024] | F16 | 1,024 |
| 176 | `blocks.2.0.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 177 | `blocks.2.0.mlp.2.bias` | [256] | F16 | 256 |
| 178 | `blocks.2.0.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 179 | `blocks.2.0.norm.bias` | [256] | F16 | 256 |
| 180 | `blocks.2.0.norm.weight` | [256] | F16 | 256 |
| 181 | `blocks.2.1.conv.bias` | [256] | F16 | 256 |
| 182 | `blocks.2.1.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 183 | `blocks.2.1.mlp.0.bias` | [1024] | F16 | 1,024 |
| 184 | `blocks.2.1.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 185 | `blocks.2.1.mlp.2.bias` | [256] | F16 | 256 |
| 186 | `blocks.2.1.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 187 | `blocks.2.1.norm.bias` | [256] | F16 | 256 |
| 188 | `blocks.2.1.norm.weight` | [256] | F16 | 256 |
| 189 | `blocks.2.2.conv.bias` | [256] | F16 | 256 |
| 190 | `blocks.2.2.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 191 | `blocks.2.2.mlp.0.bias` | [1024] | F16 | 1,024 |
| 192 | `blocks.2.2.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 193 | `blocks.2.2.mlp.2.bias` | [256] | F16 | 256 |
| 194 | `blocks.2.2.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 195 | `blocks.2.2.norm.bias` | [256] | F16 | 256 |
| 196 | `blocks.2.2.norm.weight` | [256] | F16 | 256 |
| 197 | `blocks.2.3.conv.bias` | [256] | F16 | 256 |
| 198 | `blocks.2.3.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 199 | `blocks.2.3.mlp.0.bias` | [1024] | F16 | 1,024 |
| 200 | `blocks.2.3.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 201 | `blocks.2.3.mlp.2.bias` | [256] | F16 | 256 |
| 202 | `blocks.2.3.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 203 | `blocks.2.3.norm.bias` | [256] | F16 | 256 |
| 204 | `blocks.2.3.norm.weight` | [256] | F16 | 256 |
| 205 | `blocks.2.4.conv.bias` | [256] | F16 | 256 |
| 206 | `blocks.2.4.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 207 | `blocks.2.4.mlp.0.bias` | [1024] | F16 | 1,024 |
| 208 | `blocks.2.4.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 209 | `blocks.2.4.mlp.2.bias` | [256] | F16 | 256 |
| 210 | `blocks.2.4.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 211 | `blocks.2.4.norm.bias` | [256] | F16 | 256 |
| 212 | `blocks.2.4.norm.weight` | [256] | F16 | 256 |
| 213 | `blocks.2.5.conv.bias` | [256] | F16 | 256 |
| 214 | `blocks.2.5.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 215 | `blocks.2.5.mlp.0.bias` | [1024] | F16 | 1,024 |
| 216 | `blocks.2.5.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 217 | `blocks.2.5.mlp.2.bias` | [256] | F16 | 256 |
| 218 | `blocks.2.5.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 219 | `blocks.2.5.norm.bias` | [256] | F16 | 256 |
| 220 | `blocks.2.5.norm.weight` | [256] | F16 | 256 |
| 221 | `blocks.2.6.conv.bias` | [256] | F16 | 256 |
| 222 | `blocks.2.6.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 223 | `blocks.2.6.mlp.0.bias` | [1024] | F16 | 1,024 |
| 224 | `blocks.2.6.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 225 | `blocks.2.6.mlp.2.bias` | [256] | F16 | 256 |
| 226 | `blocks.2.6.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 227 | `blocks.2.6.norm.bias` | [256] | F16 | 256 |
| 228 | `blocks.2.6.norm.weight` | [256] | F16 | 256 |
| 229 | `blocks.2.7.conv.bias` | [256] | F16 | 256 |
| 230 | `blocks.2.7.conv.weight` | [256, 3, 3, 3, 256] | F16 | 1,769,472 |
| 231 | `blocks.2.7.mlp.0.bias` | [1024] | F16 | 1,024 |
| 232 | `blocks.2.7.mlp.0.weight` | [1024, 256] | F16 | 262,144 |
| 233 | `blocks.2.7.mlp.2.bias` | [256] | F16 | 256 |
| 234 | `blocks.2.7.mlp.2.weight` | [256, 1024] | F16 | 262,144 |
| 235 | `blocks.2.7.norm.bias` | [256] | F16 | 256 |
| 236 | `blocks.2.7.norm.weight` | [256] | F16 | 256 |
| 237 | `blocks.2.8.conv1.bias` | [1024] | F16 | 1,024 |
| 238 | `blocks.2.8.conv1.weight` | [1024, 3, 3, 3, 256] | F16 | 7,077,888 |
| 239 | `blocks.2.8.conv2.bias` | [128] | F16 | 128 |
| 240 | `blocks.2.8.conv2.weight` | [128, 3, 3, 3, 128] | F16 | 442,368 |
| 241 | `blocks.2.8.norm1.bias` | [256] | F16 | 256 |
| 242 | `blocks.2.8.norm1.weight` | [256] | F16 | 256 |
| 243 | `blocks.3.0.conv.bias` | [128] | F16 | 128 |
| 244 | `blocks.3.0.conv.weight` | [128, 3, 3, 3, 128] | F16 | 442,368 |
| 245 | `blocks.3.0.mlp.0.bias` | [512] | F16 | 512 |
| 246 | `blocks.3.0.mlp.0.weight` | [512, 128] | F16 | 65,536 |
| 247 | `blocks.3.0.mlp.2.bias` | [128] | F16 | 128 |
| 248 | `blocks.3.0.mlp.2.weight` | [128, 512] | F16 | 65,536 |
| 249 | `blocks.3.0.norm.bias` | [128] | F16 | 128 |
| 250 | `blocks.3.0.norm.weight` | [128] | F16 | 128 |
| 251 | `blocks.3.1.conv.bias` | [128] | F16 | 128 |
| 252 | `blocks.3.1.conv.weight` | [128, 3, 3, 3, 128] | F16 | 442,368 |
| 253 | `blocks.3.1.mlp.0.bias` | [512] | F16 | 512 |
| 254 | `blocks.3.1.mlp.0.weight` | [512, 128] | F16 | 65,536 |
| 255 | `blocks.3.1.mlp.2.bias` | [128] | F16 | 128 |
| 256 | `blocks.3.1.mlp.2.weight` | [128, 512] | F16 | 65,536 |
| 257 | `blocks.3.1.norm.bias` | [128] | F16 | 128 |
| 258 | `blocks.3.1.norm.weight` | [128] | F16 | 128 |
| 259 | `blocks.3.2.conv.bias` | [128] | F16 | 128 |
| 260 | `blocks.3.2.conv.weight` | [128, 3, 3, 3, 128] | F16 | 442,368 |
| 261 | `blocks.3.2.mlp.0.bias` | [512] | F16 | 512 |
| 262 | `blocks.3.2.mlp.0.weight` | [512, 128] | F16 | 65,536 |
| 263 | `blocks.3.2.mlp.2.bias` | [128] | F16 | 128 |
| 264 | `blocks.3.2.mlp.2.weight` | [128, 512] | F16 | 65,536 |
| 265 | `blocks.3.2.norm.bias` | [128] | F16 | 128 |
| 266 | `blocks.3.2.norm.weight` | [128] | F16 | 128 |
| 267 | `blocks.3.3.conv.bias` | [128] | F16 | 128 |
| 268 | `blocks.3.3.conv.weight` | [128, 3, 3, 3, 128] | F16 | 442,368 |
| 269 | `blocks.3.3.mlp.0.bias` | [512] | F16 | 512 |
| 270 | `blocks.3.3.mlp.0.weight` | [512, 128] | F16 | 65,536 |
| 271 | `blocks.3.3.mlp.2.bias` | [128] | F16 | 128 |
| 272 | `blocks.3.3.mlp.2.weight` | [128, 512] | F16 | 65,536 |
| 273 | `blocks.3.3.norm.bias` | [128] | F16 | 128 |
| 274 | `blocks.3.3.norm.weight` | [128] | F16 | 128 |
| 275 | `blocks.3.4.conv1.bias` | [512] | F16 | 512 |
| 276 | `blocks.3.4.conv1.weight` | [512, 3, 3, 3, 128] | F16 | 1,769,472 |
| 277 | `blocks.3.4.conv2.bias` | [64] | F16 | 64 |
| 278 | `blocks.3.4.conv2.weight` | [64, 3, 3, 3, 64] | F16 | 110,592 |
| 279 | `blocks.3.4.norm1.bias` | [128] | F16 | 128 |
| 280 | `blocks.3.4.norm1.weight` | [128] | F16 | 128 |
| 281 | `from_latent.bias` | [1024] | F16 | 1,024 |
| 282 | `from_latent.weight` | [1024, 32] | F16 | 32,768 |
| 283 | `output_layer.bias` | [6] | F16 | 6 |
| 284 | `output_layer.weight` | [6, 64] | F16 | 384 |

