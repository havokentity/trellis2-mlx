# Open questions resolved against upstream

This document resolves every **[VERIFY]** marker in
[`PHASE0_SPEC.md`](../PHASE0_SPEC.md) and every numbered question in §8 of the
spec. Each answer cites the upstream Microsoft source (cloned into
`reference/microsoft-trellis2/`, gitignored) with `path:line` references; cite
the same way in code and PRs that act on these findings.

When this document and `PHASE0_SPEC.md` disagree, **this document wins** —
the spec was drafted from the paper and the public model card and contains
several inferences that turned out to be wrong against the real source.
Material corrections to the spec are flagged with **⚠ SPEC CORRECTION**.

Status legend: ✅ resolved · ⚠ resolved with spec correction · ❓ partially
resolved · 🔵 verify during implementation.

> Path prefixes below are relative to `reference/microsoft-trellis2/` unless
> stated otherwise.

---

## §8 numbered questions

### Q1 — RoPE 3D formulation ✅

**Per-axis interleaved RoPE with shared base 10000, applied via complex
multiplication after coordinate-driven phase computation.**

Implementation (identical for dense and sparse paths):

- Dense path: `trellis2/modules/attention/rope.py:6-48`
- Sparse path: `trellis2/modules/sparse/attention/rope.py:7-58`

Key constants (both files):

| Parameter | Value | Source |
|---|---|---|
| `dim` (spatial axes) | 3 | `rope.py:11` |
| `rope_freq` | `(1.0, 10000.0)` | `rope.py:12` |
| `freq_dim` | `head_dim // 2 // dim` = 128 // 2 // 3 = **21** | `rope.py:19` |

Phases for each axis: `freqs[k] = 1.0 / 10000**(k / 21)` for k ∈ [0, 21);
`phases[i, k] = polar(1, coord_axis[i] * freqs[k])`. Per-axis phases are
concatenated along the channel dim in **(x, y, z) order**
(`rope.py:46` — `coords.reshape(-1)` flattens [N,3] row-major, `_get_phases`
then reshapes back). The final 21×3 = 63 channels are **right-padded with
identity rotations** to reach `head_dim // 2 = 64` (`rope.py:42-47`).

The rotation itself is the standard complex-multiply formulation: features
are reshaped as complex pairs along the last dim, element-wise multiplied
by the precomputed phases, then re-viewed as real (`rope.py:29-33`).

⚠ **SPEC CORRECTION:** spec §5.9 floats "likely 10000 like standard RoPE" —
confirmed exactly. Spec §4.4 says channel layout is "interleaved per-axis,
dim split into thirds"; what upstream actually does is *contiguous blocks
per axis* (not interleaved), with the trailing slot zero-padded. Our MLX
kernel must match this contiguous block layout — interleaving silently
corrupts generation.

### Q2 — DiT FFN activation ✅

**GELU with `approximate="tanh"`. `mlp_ratio = 5.3334`, hidden FFN dim = 1536 × 5.3334 ≈ 8192.**

- Sparse FFN definition: `trellis2/modules/sparse/transformer/blocks.py:11-21`
  uses `SparseGELU(approximate="tanh")` between two linears.
- Actual `mlp_ratio` from checkpoint config: 5.3334
  (`configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json:13`,
  `configs/gen/ss_flow_img_dit_1_3B_64_bf16.json:13`, all three DiT configs).

Spec §4.4 was correct on **hidden dim 8192** but the upstream uses
`mlp_ratio` literal (5.3334) rather than an exact multiple — MLX bindings
should follow the literal.

### Q3 — DINOv3 input resolution + features ⚠

**Two resolutions: 512×512 for the SS-DiT and the 512 SLAT DiTs; 1024×1024
for the 1024 SLAT DiTs. ImageNet (not DINOv3-specific) normalization.
ALL patch tokens used as cross-attention KV (no specific layer pulled, no
CLS pooling).**

- Feature extractor: `trellis2/modules/image_feature_extractor.py:59-118`
- Pretrained checkpoint:
  `facebook/dinov3-vitl16-pretrain-lvd1689m` (configs/gen/*.json, key
  `image_cond_model.args.model_name`).
- Image preprocessing: PIL Lanczos resize to `(image_size, image_size)`,
  divide by 255, ImageNet normalize
  `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`
  (`image_feature_extractor.py:68-70, 109-110`).
- Forward pass: embeddings → RoPE embeddings → 24 transformer layers →
  final `F.layer_norm` over last hidden states; **no head, no pooling**
  (`image_feature_extractor.py:81-92`). Returns shape `(B, N, 1024)` where
  `N` includes patch + CLS + any DINOv3 register tokens.
- Pipeline calls the extractor twice at different `image_size` values:
  - `cond_512` at 512×512 (`trellis2_image_to_3d.py:539`, used by SS-DiT
    and 512 SLAT DiTs)
  - `cond_1024` at 1024×1024 (`trellis2_image_to_3d.py:540`, used by
    1024 SLAT DiTs)
- `image_size` is mutated on the same model instance per call
  (`trellis2_image_to_3d.py:174`) — DINOv3's RoPE position embeddings make
  this resolution-flexible.
- Negative (unconditional) embedding is **all-zeros**, not a separate "null"
  embedding (`trellis2_image_to_3d.py:182`).

⚠ **SPEC CORRECTION:** spec §4.1 hypothesized 224 (or possibly 518). Actual
is **512 / 1024**. Spec also says "Normalize to DINOv3 statistics"; actual
uses ImageNet statistics.

### Q4 — Sampling steps and CFG defaults per stage ⚠

**12 steps per stage; CFG 7.5/7.5/1.0; interval CFG and CFG-rescale active;
all three stages use `FlowEulerGuidanceIntervalSampler`.**

Source: `pipeline.json` (top of the HF checkpoint repo, downloaded into
`reference/weights/pipeline.json`):

| Stage | Sampler | steps | guidance_strength | guidance_rescale | guidance_interval | rescale_t |
|---|---|---|---|---|---|---|
| 1 (SS-DiT) | FlowEulerGuidanceIntervalSampler | 12 | 7.5 | 0.7 | [0.6, 1.0] | 5.0 |
| 2 (shape SLAT DiT) | FlowEulerGuidanceIntervalSampler | 12 | 7.5 | 0.5 | [0.6, 1.0] | 3.0 |
| 3 (texture SLAT DiT) | FlowEulerGuidanceIntervalSampler | 12 | 1.0 | 0.0 | [0.6, 0.9] | 3.0 |

- Sampler base implementation: `trellis2/pipelines/samplers/flow_euler.py`
- CFG mixin: `trellis2/pipelines/samplers/classifier_free_guidance_mixin.py`
- Guidance-interval mixin:
  `trellis2/pipelines/samplers/guidance_interval_mixin.py`
- `rescale_t` reparametrization (concentrates steps near `t=1`):
  `flow_euler.py:115-117` —
  `t' = rescale_t * t / (1 + (rescale_t - 1) * t)`.
- Texture stage 3 effectively turns CFG off (`guidance_strength=1.0`,
  `guidance_rescale=0.0`) — only the interval gate matters; we still need
  the CFG plumbing because the unconditional path is still computed.

⚠ **SPEC CORRECTION:** spec §2.1 says "Default 25–50 steps per stage with
classifier-free guidance (scale ~3–7.5)". Actual is **12 steps** with the
three-stage CFG profile above. We need a **CFG-rescale** implementation
(Lin et al., "Common Diffusion Noise Schedules and Sample Steps are
Flawed") and a **guidance-interval** gate, neither of which the spec
mentions. Spec also names a default scale 3.0 inside the sampler
constructor — that's a function default, overridden by the pipeline.

### Q5 — Latent grid resolution per output resolution ⚠

**Stage 1 (SS-DiT) is dense 16³ × 8 channels. SC-VAE upsamples to 64³ binary
occupancy. Pipeline max-pools to 32³ or 64³. Stage 2/3 SLAT DiTs are sparse
on a 32³ (for 512 output) or 64³ (for 1024 output) latent grid with 32
latent channels per voxel.**

Stage 1 (SS) — `configs/gen/ss_flow_img_dit_1_3B_64_bf16.json`:

```json
"resolution": 16,
"in_channels": 8,
"out_channels": 8,
"model_channels": 1536,
"num_blocks": 30,
"num_heads": 12,
"mlp_ratio": 5.3334,
"pe_mode": "rope",
"share_mod": true,
"qk_rms_norm": true
```

⚠ **SPEC CORRECTION:** spec §4.4 says stage 1 has `in_dim = 32`. Actual is
**8**. The 32-dim latent applies only to stages 2/3. Spec also says stage 1
operates on N³ "where N is small (32 or 64)". Actual is **16³**; the 64
in the checkpoint filename refers to the SS-VAE's *output* dense
resolution.

SS-VAE decoder is the legacy TRELLIS-1 dense conv decoder, reused as-is:
`microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16`
(`ss_flow_img_dit_1_3B_64_bf16.json:27`,
`pipeline.json` → `sparse_structure_decoder`). The pipeline thresholds the
decoded volume `>0` (`trellis2_image_to_3d.py:227`), max-pools to the
selected resolution (`trellis2_image_to_3d.py:230-232`), and uses the
nonzero coords as the active voxel set fed into stage 2.

Stage 2/3 (SLAT) — `configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json`
(512 variant), `configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_ft1024.json`
(1024 variant):

| Variant | `resolution` | `in_channels` | `out_channels` |
|---|---:|---:|---:|
| Shape, 512 | 32 | 32 | 32 |
| Shape, 1024 | 64 | 32 | 32 |
| Texture, 512 | 32 | **64** | 32 |
| Texture, 1024 | 64 | **64** | 32 |

Texture stage takes 64 input channels because the geometry latent is
concatenated channel-wise as conditioning (`structured_latent_flow.py:177-178`,
`trellis2_image_to_3d.py:411-419`). Output is always 32 channels.

SC-VAE upsamples latent → output at **16× spatial factor**
(four `SparseResBlockC2S3d` upsample stages, each 2× — see
`configs/scvae/shape_vae_next_dc_f16c32_fp16.json:46-78`). Hence
32³ → 512³, 64³ → 1024³. The 1024 SLAT model is a fine-tune of the 512
model on a 64³ grid (filename `_ft1024`).

Pipeline supports four modes (`trellis2_image_to_3d.py:541`):

| `pipeline_type` | SS res | SLAT res | Output res |
|---|:---:|:---:|:---:|
| `'512'` | 32 | 32 | 512 |
| `'1024'` | 64 | 64 | 1024 |
| `'1024_cascade'` (default) | 32 | 32 → 64 | 1024 |
| `'1536_cascade'` | 32 | 32 → 64 | up to 1536 |

The cascade modes run stage 2 first at 32³ with the 512 model, then
upsample coords via `shape_slat_decoder.upsample(slat, upsample_times=4)`
(`trellis2_image_to_3d.py:323`), then re-sample stage 2 at the higher
resolution with the 1024 model. A token-budget heuristic reduces
resolution if the active set grows past `max_num_tokens=49152`
(`trellis2_image_to_3d.py:328-339`).

### Q6 — Pruning mask thresholding and compaction ✅

**Predicted as a 1-layer SparseLinear → 8-dim logits per parent voxel;
threshold at logit > 0 (= sigmoid > 0.5).**

- Predictor: `trellis2/models/sc_vaes/sparse_unet_vae.py:53`
  `self.to_subdiv = sp.SparseLinear(channels, 8)`. A single linear projection,
  not an MLP.
- Threshold: `sparse_unet_vae.py:63`
  `x = self.updown(x, subdiv.replace(subdiv.feats > 0))`. Greater-than-zero
  on the raw logit (no sigmoid).
- The thresholded mask drives the sparse upsample
  `sp.SparseUpsample(2)` or `sp.SparseChannel2Spatial(2)` defined at
  `sparse_unet_vae.py:54-57`, which scatters from 1 parent to ≤8 surviving
  children. Compaction (prefix-sum on the mask) happens inside that
  SparseUpsample call.

⚠ **SPEC CORRECTION:** spec §5.5 says "a small MLP head". Actual is a
**single linear** (no hidden layer). Simpler than expected — good for our
Metal kernel.

### Q7 — Quad winding and split rule ⚠

**Per-axis 4-voxel ring uses *positive* offsets only, in CCW order. Split
diagonal is chosen by `γ` (split weight) at inference: pick diagonal with
larger product of γ at its two endpoints.**

Reference Python (the upstream calls a CUDA op at runtime; the canonical
algorithm is in this Python implementation in `o_voxel`):
`o-voxel/o_voxel/convert/flexible_dual_grid.py:142-283`

Per-axis ring offsets (rows of the table are x/y/z axes, columns are the 4
voxels around that edge, CCW as seen from +axis)
(`flexible_dual_grid.py:173-177`):

| Axis | Offsets (Δx, Δy, Δz) |
|---|---|
| **x** | (0,0,0), (0,0,1), (0,1,1), (0,1,0) |
| **y** | (0,0,0), (1,0,0), (1,0,1), (0,0,1) |
| **z** | (0,0,0), (0,1,0), (1,1,0), (1,0,0) |

So the 4 voxels around the **−axis face of voxel i** (the spec's δᵢ
convention) live at non-negative offsets from `i`. Spec §5.6 (worked
example for the X-axis edge) had it backwards — it implied negative
offsets like `(i.x-1, i.y, i.z)`. Use the positive-offset convention above.

Triangulation rules — two candidate splits
(`flexible_dual_grid.py:179-181`):

```
quad_split_1 = [0, 1, 2, 0, 2, 3]   # diagonal 0–2 → tris (0,1,2), (0,2,3)
quad_split_2 = [0, 1, 3, 3, 1, 2]   # diagonal 1–3 → tris (0,1,3), (3,1,2)
```

Inference-time selection (`flexible_dual_grid.py:257-265`):

- If `γ` (split_weight) is None → geometric fallback: pick the diagonal
  with the smaller dihedral angle between the two resulting triangles
  (i.e. flatter quad). Lines 245-256.
- If `γ` is present → compare `γ[0]*γ[2]` vs `γ[1]*γ[3]`; pick the diagonal
  whose endpoints have the larger product. Lines 258-265.

`γ` itself is per-active-voxel (not per-quad). The decoder emits it through
a **softplus** (range `(0, ∞)`), so it is a *positive scalar* with no upper
bound (not bounded in `(0, 1)` as spec §3.1 claims). See `fdg_vae.py:89`
(train) and `fdg_vae.py:102` (eval).

Training mode subdivides each quad into 4 triangles around a soft midpoint
interpolated by `γ` so the split decision is differentiable
(`flexible_dual_grid.py:266-281`). Inference uses the hard split.

⚠ **SPEC CORRECTION:** spec §3.1 says `γ ∈ (0, 1)`. Actual range is
`(0, ∞)` via softplus. Spec §5.6 ring offsets need to be replaced with the
positive-offset convention above.

### Q8 — Background removal model + transform ⚠

**Upstream uses BiRefNet (`ZhengPeng7/BiRefNet`), NOT RMBG-2.0.**

- Wrapper: `trellis2/pipelines/rembg/BiRefNet.py:1-39`
- Loaded via
  `AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)`
- Preprocessing: resize to 1024×1024, ToTensor, ImageNet normalize
  (`BiRefNet.py:13-19`).
- Forward: `model(input)[-1].sigmoid()` → single-channel probability map,
  resized back to the source size, applied as the alpha channel
  (`BiRefNet.py:32-38`).
- Pipeline integration (`trellis2_image_to_3d.py:147`,
  `trellis2_image_to_3d.py:151-162`): if input has a non-trivial alpha
  channel already, skip BiRefNet; otherwise run it, threshold the alpha
  `> 0.8 * 255` to find the bounding box, center-crop, multiply RGB by
  alpha (premultiply), and pass the result to DINOv3.

⚠ **SPEC CORRECTION:** spec §2 step 1 and §8 Q8 both say RMBG-2.0. Replace
with **BiRefNet** (`ZhengPeng7/BiRefNet`). Update
`trellis2_mlx/utils/preprocess.py` docstrings accordingly. Note that
`trust_remote_code=True` is required — BiRefNet ships its own model class
through HF's remote-code mechanism. On Apple Silicon we'll either run
BiRefNet through `transformers` on the GPU or port the segmentation
weights to MLX in a later phase. The CPU+ANE path via CoreML is a likely
optimization target.

### Q9 — Shape decoder: logits or probabilities for δ? ✅

**Raw logits at inference, thresholded at 0. Output head emits 7 channels:
3 for `v` (sigmoid + voxel margin), 3 for `δ` (raw logits), 1 for `γ`
(softplus).**

- Decoder forward: `trellis2/models/sc_vaes/fdg_vae.py:83-110`
- Inference branch (`fdg_vae.py:97-110`):

```python
vertices = (1 + 2*voxel_margin) * sigmoid(h.feats[..., 0:3]) - voxel_margin
intersected = h.replace(h.feats[..., 3:6] > 0)         # ← raw logit threshold
quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
```

- Default `voxel_margin = 0.5` (`fdg_vae.py:63`) → `v ∈ [-0.5, 1.5]`, not
  `[0, 1]`. The dual vertex is allowed to escape the cell by up to one
  voxel size in each direction; that's literally the "flexible" in
  Flexible Dual Grid.
- Training branch returns `intersected_logits` (no threshold) for BCE
  supervision (`fdg_vae.py:88`).

⚠ **SPEC CORRECTION:** spec §3.1 has `v ∈ [0, 1]`. Actual is `[-0.5, 1.5]`
(default `voxel_margin=0.5`). Spec §5.5 / §4.3 should also note that the
shape decoder's final output head is **7 channels** in the order
`(vx, vy, vz, δx, δy, δz, γ)`.

### Q10 — bf16 vs fp16 in original training ✅

**DiTs trained in bf16 (AMP); SC-VAEs trained in fp16 inflat-all mode.**

- DiT mixed-precision:
  `configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json:64-65`:
  ```json
  "mix_precision_mode": "amp",
  "mix_precision_dtype": "bfloat16"
  ```
- VAE precision:
  `configs/scvae/shape_vae_next_dc_f16c32_fp16.json:108-109`:
  ```json
  "fp16_mode": "inflat_all",
  "fp16_scale_growth": 0.001
  ```

This matches our spec policy verbatim (bf16 DiT, fp16 VAE). M4 has native
bf16; do **not** cast DiT weights to fp16 on load. VAE weights are
genuinely fp16 — when we load them into MLX, store as fp16 or cast up to
bf16 (no observable accuracy difference at inference; bf16 reduces kernel
variants).

---

## Inline `[VERIFY]` items from PHASE0_SPEC.md

### §2.1 — sampling steps and CFG defaults ✅

See **Q4** above. 12 steps / 7.5 / 7.5 / 1.0 with interval CFG and
CFG-rescale.

### §4.1 — DINOv3 input size ⚠

See **Q3**. 512×512 or 1024×1024 depending on stage; not 224, not 518.

### §4.1 — which DINOv3 layer / CLS token ✅

All layers run; the cross-attention KV is **the final layer's
LayerNorm-normalized hidden state** including all tokens (CLS, register
tokens, patch tokens). No layer skipping, no CLS pooling. See `Q3` and
`image_feature_extractor.py:81-92`.

### §4.4 — FFN activation ✅

GELU (`approximate="tanh"`), `mlp_ratio = 5.3334`. See **Q2**.

### §4.4 — RoPE 3D formulation ⚠

Per-axis contiguous blocks (not interleaved), base 10000. See **Q1**.

### §5.1 — M4 threadgroup memory limit 🔵

Apple M3/M4 Pro and Max GPUs expose **32 KB** of threadgroup memory per
threadgroup according to the Metal Feature Set Tables ("Apple7" / "Apple8"
families). To be measured empirically with `MTLDevice.maxThreadgroupMemoryLength`
once we have an MLX C++ extension building — record the actual value in a
test fixture. Spec's 32 KB working assumption is consistent with the
public docs.

### §5.7 — Trilinear: behavior for missing voxels 🔵

Upstream uses a CUDA op `flex_gemm.ops.grid_sample.grid_sample_3d`
(`o-voxel/o_voxel/postprocess.py:9`,
`trellis2/representations/mesh/base.py:5`); its source isn't in the
TRELLIS.2 repo. The end-to-end behavior in
`o-voxel/o_voxel/postprocess.py:260-289` is:

1. Trilinear-sample at every valid UV texel.
2. Texels that miss the active set are recorded in a mask.
3. Use `cv2.inpaint(..., cv2.INPAINT_TELEA)` on those mask pixels to fill
   2D gaps in UV space.

So at the *sampling* layer we likely return zero (or NaN) for missing
voxels and rely on **2D inpainting in UV space** as the recovery
mechanism. We should mirror that — our `trilinear_bake.metal` returns
zero where any of the 8 corners is missing and exposes the mask to the
texture-baking caller. Will confirm with an end-to-end run against the
reference once weights are loaded.

### §5.9 — RoPE base frequency ✅

10000. See **Q1**.

### §7.1 — Exact weight file names and key prefixes ✅

Confirmed from the HF repo + downloaded checkpoint
(`reference/weights/ckpts/`):

| File | Role |
|---|---|
| `ss_flow_img_dit_1_3B_64_bf16.safetensors` | Stage 1 SS-DiT (1.3B, bf16) |
| `slat_flow_img2shape_dit_1_3B_512_bf16.safetensors` | Stage 2 shape DiT @ 512 |
| `slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors` | Stage 2 shape DiT @ 1024 (ft) |
| `slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors` | Stage 3 texture DiT @ 512 |
| `slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors` | Stage 3 texture DiT @ 1024 (ft) |
| `shape_enc_next_dc_f16c32_fp16.safetensors` | SC-VAE shape encoder (fp16) |
| `shape_dec_next_dc_f16c32_fp16.safetensors` | SC-VAE shape decoder (fp16) |
| `tex_enc_next_dc_f16c32_fp16.safetensors` | SC-VAE material encoder (fp16) |
| `tex_dec_next_dc_f16c32_fp16.safetensors` | SC-VAE material decoder (fp16) |
| `pipeline.json` | Sampler / model wiring config |
| `texturing_pipeline.json` | Sampler / model wiring for texturing-only pipeline |

The SS decoder (`sparse_structure_decoder`) is *not* in this repo — it's
pulled from the older `microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16`
(`pipeline.json`, `models.sparse_structure_decoder`). We need to download
that separately when wiring up stage 1. Exact key prefixes per safetensors
will be enumerated in `docs/weight-inventory.md`.

### §7.2 — Linear weight layout ✅

PyTorch and MLX both store `nn.Linear` weight as `[out_features, in_features]`;
the matmul is `x @ W.T`. For weight conversion this is a **no-op transpose**
on Linear layers. We still need to verify any conv weight layouts and bias
shapes (trivial), and to remap parameter names to MLX module paths.

---

## Additional findings (not in spec §8 but material)

### Cross-attention is **not** AdaLN-modulated ⚠

The DiT block has three sub-layers — self-attn, cross-attn, FFN — but only
**two** AdaLN modulation gates (`shift_msa, scale_msa, gate_msa,
shift_mlp, scale_mlp, gate_mlp`). Cross-attention sits between them with
its own affine LayerNorm and no AdaLN gating
(`trellis2/modules/sparse/transformer/modulated.py:106-160`).

⚠ **SPEC CORRECTION:** spec §4.4 implies AdaLN-single is on both branches
of every sub-layer. Actual: AdaLN modulates self-attn and FFN; cross-attn
is plain. Update `nn/dit_block.py` accordingly.

### `share_mod` (PixArt-α style) is enabled in the pretrained checkpoint ✅

All three DiT configs have `"share_mod": true`. The shared modulation MLP
(`SiLU + Linear(C → 6C)`) lives at the model level
(`structured_latent_flow.py:54-58`,
`sparse_structure_flow.py:96-100`), is computed **once per timestep**
outside the block loop (`structured_latent_flow.py:184-187`), and each
block adds a learned per-block bias (size `6C`) before chunking. This
matches spec §4.4's description of AdaLN-single; we should follow the
shared+bias variant rather than the per-block-MLP variant.

### QK-Norm is **always enabled** in the pretrained DiTs ✅

`qk_rms_norm: true` and `qk_rms_norm_cross: true` for all three DiT configs.
Implementation uses `LayerNorm32` style at `eps=1e-6` (the same value spec
§4.4 calls out).

### Training timestep schedule ⚠

- SS-DiT: `t_schedule = logitNormal(mean=1, std=1)`
  (`configs/gen/ss_flow_img_dit_1_3B_64_bf16.json:61-67`). Matches spec.
- Both SLAT DiTs: `t_schedule = uniform`
  (`configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json:84-87`). **Spec
  §2.1 says logitNorm(1,1) for all stages — that's wrong for stages 2/3.**

Relevant for fine-tuning only; not used at inference.

### Shape latent normalization is per-channel ✅

`pipeline.json` carries `shape_slat_normalization.mean` and `.std` as
32-element vectors (one per latent channel). The pipeline subtracts mean
and divides by std before feeding into the DiT, and re-applies the
inverse on the way out (`trellis2_image_to_3d.py:271-274`,
`tex` analogue at `407-409` and `428-431`). The weight converter must
preserve these.

### Cascade pipeline tokens budget ✅

`max_num_tokens = 49152` is the hard cap on active voxels per stage
(`trellis2_image_to_3d.py:287`, `541`). If a cascade run blows past it,
the pipeline silently reduces the target resolution in 128-voxel steps
until the budget is met (`trellis2_image_to_3d.py:328-339`). Useful when
estimating memory headroom on M4 Max.

### Image conditioning is applied **with classifier-free guidance** via empty embedding ✅

`neg_cond = torch.zeros_like(cond)` (`trellis2_image_to_3d.py:182`). No
separate learned null embedding. Our pipeline should keep this convention
or generations will not match.

### Material decoder is conditioned on shape decoder's `subs` ✅

The shape decoder returns a list of `subs` (sub-structures at each
upsampling stage); the material decoder takes them as `guide_subs`
(`trellis2_image_to_3d.py:450`). This is how the material decoder
inherits the shape decoder's pruning decisions at every level — exactly
what spec §4.3 describes.

---

## Items still open / out-of-band

- **Exact DINOv3 token count per resolution**: needs a one-line probe
  against `transformers.DINOv3ViTModel` at the chosen `image_size`. Not
  blocking — we'll get the number when we run the encoder in Phase 1
  step 3.
- **`flex_gemm.ops.grid_sample` exact signature**: we have its call site
  but not its body. Replicate based on PyTorch's `grid_sample`
  conventions and verify numerically against an end-to-end reference run
  during Phase 1 step 10.
- **Exact submanifold conv weight layout per channel order**: upstream
  has three back-ends (`spconv`, `torchsparse`, `flex_gemm`) — the
  shipped checkpoint was trained with one of them. The published
  safetensors will tell us the shape; the kernel offset ordering (z-y-x
  vs x-y-z) needs to be matched. Capture in
  `docs/weight-inventory.md` after the download finishes.

Last reviewed: 2026-05-13.
