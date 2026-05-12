# TRELLIS.2 → MLX Port: Phase 0 Architecture Spec

**Status:** Phase 0 — pre-implementation reference document
**Target hardware:** Apple Silicon M4 Max (≥36GB unified memory recommended)
**Target framework:** MLX + custom Metal Shading Language kernels
**Scope:** Inference + fine-tuning capability
**Source:** arxiv 2512.14692 (Xiang et al., Dec 2025), microsoft/TRELLIS.2

This document is the source-of-truth reference for everything we build. It is derived from the paper, the public source layout, and the model card. Items marked **[VERIFY]** must be confirmed against the actual upstream source before implementation.

---

## 1. Project Overview

### 1.1 What we are building

A from-scratch reimplementation of TRELLIS.2 image-to-3D generation on Apple Silicon using MLX as the host framework and custom Metal compute kernels for performance-critical sparse operations. The objective is **native GPU performance** — not a translation layer over PyTorch MPS.

### 1.2 Performance targets

| Resolution | Output | H100 reference | M4 Max realistic target |
|---|---|---|---|
| 512³ | ~2.2K latents, GLB | 3 s | 30–45 s |
| 1024³ | ~9.6K latents, GLB | 17 s | 90–180 s |
| 1536³ | cascaded | 60 s | 5–10 min |

Ceiling is set by hardware: M4 Max GPU is ~14–16 TFLOPS fp16/bf16 vs H100's ~750+. Target is to saturate the available compute and memory bandwidth, not to match H100.

### 1.3 Out of scope (initial)

- Differentiable rasterization (`nvdiffrast`). Only needed for VAE stage-2 rendering loss during training, and for the MP4 visualization. GLB export does not need it.
- `nvdiffrec` split-sum PBR renderer. Only for visualization. GLB has its own material channels.
- Training the SC-VAE from scratch. We load Microsoft's pretrained weights. Fine-tuning the DiTs is in scope; retraining the VAE is not.
- Texturing-only pipeline (`Trellis2TexturingPipeline`). Image-to-3D first.

---

## 2. End-to-End Inference Pipeline

```
Input PIL image
    │
    ▼
[1] Image preprocessing
    │  • RMBG-2.0 background removal → alpha mask
    │  • Center-crop, resize to 512 or 1024
    │  • Normalize to DINOv3 statistics
    ▼
[2] DINOv3-L image encoder (frozen)
    │  • vitl16, 16×16 patches
    │  • Output: image conditioning tokens (used by all 3 DiTs in cross-attn)
    ▼
[3] Stage 1: Sparse Structure DiT
    │  • Operates on full latent grid (32³ for 512 output, 64³ for 1024)
    │  • Predicts binary occupancy: which voxels are active
    │  • Output: active voxel coordinate set P = {pᵢ ∈ [0,N)³}
    ▼
[4] Stage 2: Geometry DiT (sparse)
    │  • Operates only on active voxels from stage 1
    │  • Predicts geometry latents zᵍ ∈ ℝ^(L × 32)
    │  • Conditioning: image features (cross-attn) + timestep (AdaLN)
    ▼
[5] Stage 3: Material DiT (sparse)
    │  • Same active voxel set
    │  • Predicts material latents zᵐ ∈ ℝ^(L × 32)
    │  • Conditioning: image features + zᵍ concatenated channel-wise (input dim 64)
    ▼
[6] SC-VAE Shape decoder
    │  • Sparse-conv U-net, 16× spatial upsampling
    │  • Input: zᵍ at coarse resolution → O-Voxel shape features at fine resolution
    │  • Early-pruning at each upsample: predicts which child voxels survive
    │  • Output: per-active-voxel (v, δ, γ) at output resolution (512³ / 1024³)
    ▼
[7] SC-VAE Material decoder
    │  • Same architecture, conditioned on shape's pruning structure
    │  • Output: per-active-voxel (c, m, r, α)
    ▼
[8] Flexible Dual Grid → mesh extraction
    │  • For each active voxel: dual vertex v positioned within voxel
    │  • Connect 4 dual vertices across each active edge δ → quad
    │  • Adaptive split each quad into 2 triangles using γ
    ▼
[9] Material baking
    │  • For each mesh vertex (or texture texel): trilinear interp of (c,m,r,α)
    │    from neighboring voxels
    ▼
GLB export with PBR materials
```

### 2.1 Sampling

All three DiT stages use **rectified flow** sampling. Forward process: x(t) = (1−t)x₀ + tε. Reverse: integrate the predicted velocity field vθ from t=1 (noise) to t=0 (data). Default 25–50 steps per stage with classifier-free guidance (scale ~3–7.5). **[VERIFY exact step count and CFG scale defaults from upstream]**

Timestep sampling during training uses logitNorm(1,1) per the paper. Not relevant for inference but matters for fine-tuning.

---

## 3. O-Voxel Data Structure

### 3.1 Per-active-voxel fields

| Field | Symbol | Type | Shape | Range | Notes |
|---|---|---|---|---|---|
| Position | p | uint16 | [3] | [0, N) | Grid coord; N ∈ {512, 1024, 1536} |
| Dual vertex | v | float | [3] | [0, 1] | Position within voxel cell |
| Edge flags | δ | uint8 (bitfield) | [3 bits] | {0,1}³ | Active edges on −X, −Y, −Z faces |
| Split weight | γ | float | [1] | (0, 1) | Quad → triangle split heuristic |
| Base color | c | float | [3] | [0, 1] | Linear-space RGB |
| Metallic | m | float | [1] | [0, 1] | |
| Roughness | r | float | [1] | [0, 1] | |
| Opacity | α | float | [1] | [0, 1] | Translucency support |

**Convention:** δᵢ refers to the voxel edge along axis i (X, Y, or Z) emanating from the minimum-coordinate corner of the cell. The other 9 edges' flags belong to the neighboring voxels — **do not duplicate**.

### 3.2 Memory layout (MLX-side)

Store O-Voxel as a Structure-of-Arrays:

```python
class OVoxel:
    coords:     mx.array  # [L, 3] int32       (we use int32 not uint16 for MLX)
    v:          mx.array  # [L, 3] float
    delta:      mx.array  # [L, 3] uint8 or float for differentiable training
    gamma:      mx.array  # [L]    float
    c:          mx.array  # [L, 3] float
    m:          mx.array  # [L]    float
    r:          mx.array  # [L]    float
    alpha:      mx.array  # [L]    float
    resolution: int       # N
```

L = number of active voxels (~9.6K for 1024³).

### 3.3 Auxiliary structures we need to maintain

| Structure | Purpose | Lifetime |
|---|---|---|
| Spatial hash `H: coord → voxel_index` | O(1) neighbor lookup for sparse conv | Recomputed per resolution change |
| Neighbor table `N: [L, 27] int32` | Submanifold conv 3×3×3 kernel offsets, −1 if absent | Recomputed per active-set change |
| Child→parent map | 8 fine voxels → 1 coarse voxel, for VAE down/up | Per VAE stage |
| Pruning mask `ρ ∈ {0,1}^(L × 8)` | Predicted per upsampling step | Per upsample call |

The neighbor table is the hot path. For 9.6K voxels × 27 neighbors = ~260K int32 = 1MB. Fits easily in L2 cache of M4 Max GPU.

---

## 4. Network Architectures

### 4.1 DINOv3-L Image Encoder

- Architecture: ViT-L/16 (24 layers, 1024 dim, 16 heads, patch 16)
- Input: 224×224 (per DINOv3 default) **[VERIFY input size for TRELLIS — may be 518 like DINOv2]**
- Pretrained checkpoint: `facebook/dinov3-vitl16-pretrain-lvd1689m`
- Output: per-patch tokens used for cross-attention conditioning. **[VERIFY which layer's features and whether CLS token is included]**
- **Frozen during all training and inference**

In MLX: this is a vanilla ViT. Use MLX's existing transformer primitives. No custom Metal needed.

**Optimization opportunity:** convert DINOv3 to CoreML and run on the Apple Neural Engine. The ViT is a static graph and a strong ANE candidate. This runs in parallel with model load and frees the GPU.

### 4.2 SC-VAE Encoder (354M params)

Used during **training** to construct latent targets. **Not needed for inference** if we use Microsoft's checkpoint as-is. Listed here for completeness and for fine-tuning support.

| Stage | f_down | Channels (in→out) | Block layout | # blocks |
|---|---|---|---|---|
| 0 | 1× | 6 → 64 | Linear | 1 |
| 0→1 | 1×→2× | 64 → 128 | ResEnc (sparse residual AE) | 1 |
| 1 | 2× | 128 → 128 | [SubMConv3×3×3 → LN → Linear(128,512) → SiLU → Linear(512,128)] | 4 |
| 1→2 | 2×→4× | 128 → 256 | ResEnc | 1 |
| 2 | 4× | 256 → 256 | [SubMConv3 → LN → Linear(256,1024) → SiLU → Linear(1024,256)] | 8 |
| 2→3 | 4×→8× | 256 → 512 | ResEnc | 1 |
| 3 | 8× | 512 → 512 | [SubMConv3 → LN → Linear(512,2048) → SiLU → Linear(2048,512)] | 16 |
| 3→4 | 8×→16× | 512 → 1024 | ResEnc | 1 |
| 4 | 16× | 1024 → 1024 | [SubMConv3 → LN → Linear(1024,4096) → SiLU → Linear(4096,1024)] | 4 |
| out | 16× | 1024 → 64 | Linear → mean+logvar split | 1 |

The "ResEnc" block is the sparse residual autoencoding layer:
1. Channel-wise group-average of the 8 children's features (provides a shortcut from fine to coarse).
2. Add to the output of a SubMConv3 + LayerNorm + Linear sequence.

Final latent: 32 channels per voxel (after KL reparam split).

### 4.3 SC-VAE Decoder (474M params)

Mirror structure with three differences:
1. ResEnc layers become **ResDec**: instead of group-averaging children, they unstack channels into 8 children and `dup_groups` to match the target dim.
2. Each upsample step is preceded by an **early-pruning** predictor: a small MLP head outputs ρ ∈ {0,1}^8 for each parent, deciding which children survive.
3. Final projection produces (v, δ, γ) for shape decoder or (c, m, r, α) for material decoder.

The material decoder is **conditioned on the shape decoder's pruning structure** — it receives the same active-set decisions, so geometry and material are spatially aligned by construction.

### 4.4 DiT Generators (3 stages, each ~1.3B params)

Identical block structure for all three stages:

```
x → InProj(in_dim → 1536)
  → 30 × DiTBlock
  → LayerNorm
  → OutProj(1536 → 32)
```

`in_dim` is:
- Stage 1 (sparse structure): 32 — but operates on **dense** N³ grid where N is small (32 or 64)
- Stage 2 (geometry): 32 — operates on sparse voxels from stage 1
- Stage 3 (material): 32+32=64 — concatenates shape latent as condition

Each `DiTBlock`:

```
[AdaLN-single]  ─── timestep → γ₁, β₁, α₁, γ₂, β₂, α₂ (6 scalars per token)
   │
   ├── SelfAttn(12 heads × 128 dim)
   │     • QK-Norm: RMSNorm on Q and K
   │     • RoPE 3D on Q, K (positional)
   │     • Modulated by γ₁, β₁, α₁
   │
   ├── LayerNorm
   ├── CrossAttn(12 heads × 128 dim, KV from DINOv3 features)
   │     • Q from token stream, KV from image features
   │     • No RoPE (image features are unordered)
   │
   ├── [AdaLN-single second branch]
   └── FFN(1536 → 8192 → 1536, GELU or SiLU [VERIFY])
         • Modulated by γ₂, β₂, α₂
```

**AdaLN-single** ([chen2024pixart]): a *shared* MLP predicts the modulation parameters from the timestep embedding, then per-layer learned scale/shift parameters are added. Drastically reduces parameters vs vanilla AdaLN-Zero. Implementation detail: the shared parameters are computed once outside the block loop; only the per-layer trainable adapters are inside.

**QK-Norm:** apply RMSNorm to Q and K **before** the scaled dot-product. Stabilizes training of large models. Use ε=1e-6.

**RoPE 3D:** rotary embedding extended to 3D coordinates. **[VERIFY]** the exact formulation — likely interleaved per-axis: dim split into thirds, each third rotates by frequency derived from the corresponding axis coordinate. Critical to verify because RoPE base frequency and axis ordering must match the checkpoint exactly or generation collapses.

---

## 5. Operations Inventory (What We Need Metal Kernels For)

Sorted by impact on inference time. Forward AND backward required for fine-tuning support.

### 5.1 Fused sparse attention (HIGHEST PRIORITY)

**Why:** Dominates inference time. ~80% of stage 2 and stage 3 wall-clock.

**Spec:**
- Input: Q ∈ [L, H, D], K ∈ [L, H, D], V ∈ [L, H, D], optional mask
- H = 12 heads, D = 128 head dim
- L can be up to ~9.6K
- Operation: standard self-attention with QK-Norm and RoPE applied before
- Output: [L, H, D]

**Strategy:** Flash-attention-style tiled algorithm with online softmax. Tile Q into blocks of 64–128 rows, K/V into blocks of 64. Per Q-block:
1. Load Q tile into threadgroup memory
2. Iterate K/V tiles: compute QK^T, online softmax update (max, sum), accumulate weighted V
3. Write final O tile

For M4 Max: threadgroup memory is 32KB per threadgroup. A Q-tile of 64×128 fp16 = 16KB. K-tile of 64×128 fp16 = 16KB. Total 32KB — fits exactly. **[VERIFY M4 threadgroup limit]**

**Backward:** Standard flash-attention backward — recompute attention in forward direction, accumulate gradients via two passes (one for dQ, one for dK/dV). Roughly 2× forward cost.

**Note:** This is *dense* attention even though tokens come from a sparse voxel grid — there's no structure to exploit at the L=9.6K scale (the attention matrix is small enough). Sparse attention patterns are NOT a win here; flash-attention dense is.

### 5.2 Submanifold sparse 3D convolution

**Why:** Every VAE encoder/decoder block uses it. ~5–10% of inference.

**Spec:**
- Input: features X ∈ [L, C_in], coords ∈ [L, 3], kernel weights W ∈ [27, C_in, C_out]
- "Submanifold" = output voxel set = input voxel set (no expansion)
- For each output voxel i: y_i = Σ_k W_k @ x_{N(i,k)} for valid neighbors N(i,k)
- Output: [L, C_out]

**Strategy — Masked Implicit GEMM** (same as upstream FlexGEMM):
1. Build neighbor table N ∈ [L, 27] (cached, computed once per active set)
2. For each kernel position k ∈ [0, 27): gather rows of X according to N[:, k], multiply by W_k, accumulate into Y
3. Mask out invalid (−1) neighbors via predicated stores

The 27 partial GEMMs can be done as ONE large GEMM with a permutation, fusing the gather into the matmul's index calculation. This is the "implicit" in implicit GEMM.

**Backward:**
- dX: scatter-add the dY @ W_k^T contributions back to neighbor positions
- dW: per kernel position, sum outer products of gathered X rows with dY rows

### 5.3 Neighbor table construction

**Why:** Built once per active set. Fast enough on CPU for small voxel counts, but must be on GPU for high resolutions and to keep buffers off CPU↔GPU bus.

**Spec:**
- Input: coords ∈ [L, 3]
- Output: spatial hash + neighbor table N ∈ [L, 27]

**Strategy:** Parallel open-addressing hashmap.
1. Allocate hash table of size 2L (load factor 0.5)
2. Kernel A: parallel insert each coord with linear-probing on conflict (use `atomic_compare_exchange`)
3. Kernel B: for each voxel i and kernel offset k (27 total): hash-lookup `coords[i] + offset[k]`, write index or −1

**Backward:** Not needed — it's a discrete index structure, not differentiable.

### 5.4 Sparse residual autoencoding (down/up)

**Why:** Used between every VAE stage. Cheap individually but needed correctly.

**Down (8 children → 1 parent):**
1. For each fine voxel, compute parent coord = coord >> 1
2. Group-by parent, stack 8 children's features → 8C channels (pad missing with zeros)
3. `avg_groups`: reshape (8, C) → (G, 8C/G) → mean over the 8-dim → (C',) where G = 8C / C'

**Up (1 parent → 8 children):**
1. `unstack`: reshape (C') → (8, C'/8) (one C'/8 vector per child)
2. `dup_groups`: tile within each group to reach target C channels
3. Add to the convolutional path

**Backward:** All linear ops; autograd handles.

### 5.5 Early-pruning predictor + active-set update

**Why:** Critical for memory — without pruning, 1024³ blows past M4 Max memory.

**Spec:**
- Input: features X ∈ [L_coarse, C], coarse coords
- Output: per-parent 8-bit mask ρ ∈ [L_coarse, 8] ∈ {0,1}, plus new fine active set
- The mask is **predicted by a small MLP head**; loss during training is BCE against the ground-truth fine activity pattern

At inference: predicted ρ is *thresholded* (e.g. at 0.5) and the corresponding fine voxels are kept. Compaction is needed afterward to produce a tight [L_fine, ...] tensor.

**Compaction kernel:** Parallel prefix-sum on the flattened mask to compute output indices, then scatter.

### 5.6 Flexible Dual Grid mesh extraction

**Why:** Called once per generation. Current pure-Python implementation is the slowest non-attention step in trellis-mac (~30–60s).

**Spec:**
- Input: O-Voxel with (coords, v, δ, γ)
- Output: vertex array V ∈ [L, 3] (one dual vertex per active voxel), face array F ∈ [M, 3] where M ≈ 2 × (# active δ-flags × valid quads)

**Algorithm:**
```
V = positions of all dual vertices (coords + v) × (1/N)
F = []
hash = build_spatial_hash(coords)
for each voxel i:
    for axis a in {X, Y, Z}:
        if not δ_i[a]: continue
        # Find the 4 voxels around this edge
        # For axis X edge at the −X face of voxel i:
        # neighbors are (i.x-1, i.y, i.z), (i.x, i.y, i.z), (i.x, i.y-1, i.z), (i.x-1, i.y-1, i.z) [VERIFY exact pattern]
        quad = [hash.lookup(c) for c in get_quad_coords(coord_i, axis_a)]
        if any quad index is missing: continue
        # Split quad into 2 triangles using γ_i (or some combination of γ's)
        t1, t2 = split_quad(quad, γ_i)
        F.append(t1, t2)
```

**Metal implementation:**
- Kernel 1: one thread per voxel, three iterations (one per axis), atomic append to F
- Use atomic counter to allocate face slots
- Worst case F size: L × 3 × 2 = 6L triangles. Pre-allocate that.

**Backward:** Differentiable rasterization is out of scope for this kernel. For fine-tuning, the dual vertex positions are supervised directly (MSE), not through mesh extraction.

### 5.7 Material trilinear interpolation (texture baking)

**Why:** Final step before GLB export. Cheap but must be correct.

**Spec:**
- Input: O-Voxel material features (c, m, r, α) ∈ [L, 6], query points Q ∈ [Nq, 3] in world coords
- Output: per-query (c, m, r, α) ∈ [Nq, 6]
- For each query: find 8 neighboring active voxels, trilinear-interpolate (missing voxels = nearest extrapolation or zero — **[VERIFY upstream behavior]**)

**Metal implementation:** straightforward — one thread per query, 8 neighbor lookups via hash, weighted sum.

### 5.8 AdaLN-single modulation

Trivial elementwise op: `(1 + γ) * LayerNorm(x) + β`. Use MLX primitives — no custom kernel.

### 5.9 RoPE 3D

Elementwise. Per-token: split channel dim into 3 groups (X, Y, Z), apply 2D rotation to each pair within a group using frequency = base^(-2k/D) and angle = coord_axis × frequency. Use MLX primitives.

**[VERIFY base frequency — paper doesn't state; likely 10000 like standard RoPE]**

---

## 6. MLX Module Plan

### 6.1 Package layout

```
trellis2_mlx/
├── __init__.py
├── pipeline.py              # Trellis2ImageTo3DPipeline (MLX)
├── ovoxel/
│   ├── data.py              # OVoxel class, hash, neighbor table
│   ├── mesh_extract.py      # Python wrapper over Metal mesh kernel
│   ├── material_bake.py     # Python wrapper over Metal interp kernel
│   └── postprocess.py       # GLB export via trimesh
├── nn/
│   ├── sparse_conv.py       # SubMConv3 module (calls Metal kernel)
│   ├── sparse_attn.py       # Sparse self-attention module
│   ├── adaln.py             # AdaLN-single
│   ├── rope.py              # 3D rotary embeddings
│   ├── dit_block.py         # The full DiT block
│   └── res_ae.py            # ResEnc / ResDec layers
├── models/
│   ├── dinov3.py            # ViT-L/16 in MLX (or CoreML wrapper)
│   ├── vae.py               # SC-VAE encoder + decoder
│   └── dit.py               # Stage1/2/3 DiT generators
├── samplers/
│   └── rectified_flow.py    # Rectified-flow sampler with CFG
├── metal/
│   ├── kernels/
│   │   ├── sparse_attn_fwd.metal
│   │   ├── sparse_attn_bwd.metal
│   │   ├── sparse_conv_fwd.metal
│   │   ├── sparse_conv_bwd.metal
│   │   ├── neighbor_build.metal
│   │   ├── prefix_sum.metal
│   │   ├── mesh_extract.metal
│   │   └── trilinear_bake.metal
│   └── ops.py               # MLX custom-op bindings
├── utils/
│   ├── preprocess.py        # Image preproc, RMBG-2.0 wrapper
│   └── weight_convert.py    # HF checkpoint → MLX state dict
└── tests/
    └── …
```

### 6.2 Custom-op binding strategy

MLX exposes custom Metal kernels via `mx.fast.metal_kernel(...)` for forward-only ops, or via the C++ extension interface for ops with autograd. Since we need backward for sparse_conv and sparse_attn, those go through the C++ extension path. mesh_extract and neighbor_build are forward-only.

**Reference:** the MLX `fast` module pattern in the MLX repo is the template — particularly `scaled_dot_product_attention` and the `examples/extensions` directory.

### 6.3 dtype policy

- **Weights:** bf16 in storage (matches model training precision)
- **Activations:** bf16 throughout the transformer, fp16 in VAE convs (more dynamic range needed for residual paths)
- **Accumulators in attention/conv:** fp32 (online softmax stats, GEMM accumulators)
- **Coordinates:** int32
- **Hash table entries:** int32 + int32 (key, value)

M4 hardware supports bf16 natively — no emulation cost.

---

## 7. Weight Conversion

### 7.1 Source format

Microsoft publishes weights at `huggingface.co/microsoft/TRELLIS.2-4B` as PyTorch `.safetensors` files. Expected structure based on the architecture:

- `image_encoder/` — DINOv3-L weights (or just reference; may be loaded directly from facebook/dinov3-vitl16)
- `vae_shape/` — SC-VAE shape encoder + decoder
- `vae_material/` — SC-VAE material encoder + decoder
- `dit_structure/` — stage 1 sparse-structure DiT
- `dit_geometry/` — stage 2 geometry DiT
- `dit_material/` — stage 3 material DiT

**[VERIFY exact file names and key prefixes from upstream]**

### 7.2 Conversion approach

Write `weight_convert.py` that:
1. Loads PyTorch state dict via `safetensors`
2. Walks the module tree of our MLX model
3. Maps each MLX parameter to a PyTorch tensor by name (using a regex/lookup table)
4. Transposes Linear weights (PyTorch is [out, in]; MLX is [out, in] also, so this may be a no-op — **[VERIFY]**)
5. Saves as MLX `.safetensors`

This is mechanical work — straightforward once we know the exact key names, which we get by `safetensors_inspect` on the first downloaded checkpoint.

---

## 8. Open Questions to Resolve Before Phase 1

Each must be verified by reading the upstream source. None block Phase 1 architecture work, but each may invalidate small parts of this spec.

| # | Question | How to verify |
|---|---|---|
| 1 | Exact RoPE 3D formulation: per-axis frequency, base, axis ordering | `trellis2/modules/attention.py` or similar |
| 2 | DiT FFN activation: GELU vs SiLU vs SwiGLU | DiT block source |
| 3 | DINOv3 input resolution and which features are used | `pipelines/image_to_3d.py` |
| 4 | Number of sampling steps and CFG scale defaults per stage | pipeline `run()` method |
| 5 | Latent grid resolution for each output res (32³ vs 64³ vs …) | VAE config |
| 6 | Exact pruning mask thresholding and compaction details | VAE decoder |
| 7 | Quad-orientation convention in mesh extraction (winding order) | `o_voxel/cumesh` source |
| 8 | Exact RMBG-2.0 version and preprocessing transform | pipeline preprocess |
| 9 | Whether shape decoder predicts logits or probabilities for δ | VAE decoder output head |
| 10 | bf16 vs fp16 in original training (affects expected numerical ranges) | training config |

---

## 9. Phase 1 — Next Concrete Steps

Once this spec is approved, work order:

1. **Set up project skeleton** — package layout per §6.1, with empty stubs and CI hooked up.
2. **Download checkpoint, inspect, write weight inventory** — produces the exact key names that resolve open questions 1–10 (many of them).
3. **DINOv3-L in MLX** — pure transformer, no custom Metal. Tests against PyTorch reference output on a sample image.
4. **OVoxel data class + neighbor-build Metal kernel** — foundation everything else sits on.
5. **SubMConv3 forward Metal kernel** — second-most-used op. Test against PyTorch spconv on synthetic sparse data.
6. **First end-to-end smoke test:** SC-VAE decoder alone, with random latent in, mesh out (no DiTs yet). Verifies sparse_conv, residual AE, pruning, and mesh extraction all integrate correctly. Quality won't be good but the pipeline runs.
7. **Sparse attention forward Metal kernel** — the big one.
8. **DiT block in MLX, wire end-to-end** — stage 2 only first (geometry), with stage 1's active set hand-computed from a reference run.
9. **Stage 1 sparse-structure DiT** — closes the loop.
10. **Stage 3 material DiT** — full pipeline.
11. **Backward kernels for sparse_conv and sparse_attn** — gates fine-tuning.
12. **Profiling pass, optimization sweep** — Xcode Metal frame capture, kernel-by-kernel analysis.

---

## 10. References

- Paper: Xiang et al., "Native and Compact Structured Latents for 3D Generation", arxiv 2512.14692
- Repo: github.com/microsoft/TRELLIS.2
- Model: huggingface.co/microsoft/TRELLIS.2-4B
- DINOv3: huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
- MLX docs: ml-explore.github.io/mlx
- Flash-Attention paper: Dao et al., "FlashAttention-2", arxiv 2307.08691
- Submanifold sparse conv: Graham et al., arxiv 1706.01307
- Dual Contouring: Ju et al., "Dual Contouring of Hermite Data", SIGGRAPH 2002
- ConvNeXt: Liu et al., arxiv 2201.03545
- DC-AE residual autoencoding: Chen et al., arxiv 2410.10733 (deep compression autoencoders)
- Rectified Flow: Liu et al., arxiv 2209.03003
- AdaLN-single: Chen et al., "PixArt-α", arxiv 2310.00426
