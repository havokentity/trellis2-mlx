# trellis2-mlx — Phase Playbook

Complete set of kickoff prompts for Claude Code, one per phase. Each phase
is a meaningful unit of work (~1–2 weeks of focused effort).

**How to use:** When starting a new phase, paste the prompt between the
`╔═══` and `╚═══` markers into Claude Code. Don't paste the surrounding
context/notes — those are for you.

**Phase order is strict.** Don't start Phase N+1 until Phase N is done and
its end-of-phase summary has been reviewed.

---

## Phase Overview

| # | Phase | Rough Time | Gate |
|---|---|---|---|
| 0 | Architecture spec | ✅ done | PHASE0_SPEC.md exists |
| 1 | Discovery & Bootstrap | 3–5 days | All §8 questions answered, weight inventory done |
| 2 | DINOv3-L + Weight Converter | 1 week | Image encoder matches PyTorch ref bit-for-bit (bf16 tol) |
| 3 | O-Voxel + Sparse Conv | 2 weeks | SubMConv3 forward matches PyTorch spconv reference |
| 4 | VAE Smoke Test | 1–2 weeks | Random latent → valid mesh exported as GLB |
| 5 | Sparse Attention | 2–3 weeks | Flash-attention-style MSL kernel passes correctness + perf bar |
| 6 | DiT + Full Pipeline | 2 weeks | End-to-end image-to-3D produces valid GLB |
| 7 | Backward Kernels | 1–2 weeks | Fine-tuning loop runs end-to-end on a tiny dataset |
| 8 | Optimization | 1–2 weeks | M4 Max hits 512³ in ≤45s, 1024³ in ≤180s |
| 9 | Release | 1 week | Public release with docs, benchmarks, example notebooks |

**Total: 12–18 weeks** at a real working pace.

---

# Phase 1: Discovery & Bootstrap

**Goal:** Establish the project foundation and resolve every architectural
ambiguity before any model code is written.

**Prerequisite:** `PHASE0_SPEC.md` is in the project root.

╔═══════════════════════════ PHASE 1 PROMPT ═══════════════════════════

I'm starting a new project: trellis2-mlx, a from-scratch Apple Silicon
port of Microsoft TRELLIS.2 (image-to-3D generation) using MLX with
custom Metal Shading Language kernels for native GPU performance.

PROJECT LOCATION
You're already in the target directory:
/Volumes/XTRM 5 Media/More MyRepos/trellis2-mlx

SOURCE OF TRUTH
PHASE0_SPEC.md is in this directory. Read it first and treat it as the
authoritative architecture reference. It covers the O-Voxel data structure
(§3), SC-VAE encoder/decoder and the three DiT generators (§4), every
Metal kernel we need to write prioritized (§5), MLX module layout (§6),
open questions that must be verified against upstream source (§8), and
the Phase 1 work order (§9). If anything I say below conflicts with the
spec, the spec wins.

WHAT TO DO IN THIS SESSION
1. Read PHASE0_SPEC.md end to end.
2. Create the project skeleton from §6.1: pyproject.toml (mlx,
   safetensors, huggingface-hub, Pillow, trimesh, transformers for RMBG,
   einops, numpy; dev extras: pytest, ruff, mypy, torch for CPU reference
   only), MIT LICENSE, README.md, CLAUDE.md with our working agreements,
   .gitignore (Python + macOS + Xcode + safetensors/glb/mp4),
   .gitattributes, and stub Python files for every module in §6.1 with
   docstrings pointing to the relevant spec section.
3. Initialize git, make an initial commit "chore: project skeleton".
4. Create a public GitHub repo named trellis2-mlx and push:
   gh repo create trellis2-mlx --public --source=. --remote=origin --push
5. Clone the upstream TRELLIS.2 source into reference/ (gitignored):
   git clone https://github.com/microsoft/TRELLIS.2.git reference/microsoft-trellis2
6. Walk through every [VERIFY] item and every numbered question in §8 of
   the spec. For each one, find the answer in the upstream source and
   record it in docs/open-questions-resolved.md with file:line citations.
7. Download the TRELLIS.2-4B checkpoint and inventory it:
   python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/TRELLIS.2-4B', local_dir='reference/weights')"
   Then walk the safetensors files and write every parameter name + shape
   to docs/weight-inventory.md. This is the source-of-truth for the
   weight converter.
8. Commit each piece. Stop when steps 1-7 are done — do NOT start writing
   model code in this session.

WORKING AGREEMENTS (apply to all future phases too)
- Apple Silicon only. MLX is the host framework. No PyTorch MPS, no
  xformers, no flash-attn, no spconv, no triton.
- PyTorch is allowed only as a CPU reference for numerical tests.
- bf16 for weights and activations, fp32 for accumulators in attention
  and conv kernels. M4 has native bf16; don't downgrade to fp16.
- Fine-tuning is in scope, so every custom op needs a backward (vjp).
- Type-hint everything. Docstring every public function. Conventional
  commit prefixes (feat:, fix:, refactor:, docs:, test:, perf:, chore:).
- Do NOT fork shivampkumar/trellis-mac. This is from-scratch.
- Do NOT invent architectural details. If something is ambiguous after
  reasonable searching, stop and ask me.

WHEN TO ASK ME
- Any [VERIFY] item where you can't find a definitive answer.
- A design choice not covered by the spec.
- A new dependency that isn't in the plan.

End-of-session summary I want: what got done, what's blocked, next step.

╚══════════════════════════════════════════════════════════════════════

**Done when:** `docs/open-questions-resolved.md` and `docs/weight-inventory.md`
exist, GitHub repo is live, all skeleton files committed.

---

# Phase 2: DINOv3-L Image Encoder + Weight Converter Foundation

**Goal:** Get the frozen image encoder running in MLX with numerical
equivalence to upstream within bf16 tolerance. Establishes the
weight-conversion pattern reused for every later module.

**Prerequisite:** Phase 1 done. You have `docs/weight-inventory.md` and
`docs/open-questions-resolved.md` answering DINOv3 input resolution and
which features are used.

╔═══════════════════════════ PHASE 2 PROMPT ═══════════════════════════

Continuing the trellis2-mlx project. Phase 1 (Discovery & Bootstrap) is
complete. Read PHASE0_SPEC.md if you haven't this session, then proceed.

Phase 2 goal: implement the DINOv3-L image encoder in MLX and establish
the weight-conversion pattern.

DELIVERABLES
1. trellis2_mlx/models/dinov3.py — ViT-L/16 in MLX (24 layers, 1024 dim,
   16 heads, patch 16). Use MLX's nn.MultiHeadAttention where it's a
   good fit; vanilla matmul where it isn't. Pure MLX, no custom Metal
   kernels in this phase.
2. trellis2_mlx/utils/weight_convert.py — generic safetensors → MLX
   state-dict converter, with a key-name mapping table. Make the
   mapping declarative (a dict or YAML) so we can reuse the pattern
   for VAE and DiT in later phases.
3. tests/test_dinov3.py — load the upstream DINOv3-L checkpoint, run
   the same input through PyTorch (CPU) and through our MLX
   implementation, compare per-layer activations and final output.
   Tolerance: 5e-3 absolute, 1e-2 relative in bf16. Tighter in fp32.
4. A demo script scripts/demo_dinov3.py that loads an image, runs the
   encoder, prints output shape and a few stats.

OPTIONAL (do only if Phase 2 finishes early)
5. CoreML conversion script scripts/dinov3_to_coreml.py — convert
   DINOv3-L to a .mlpackage runnable on the Apple Neural Engine. This
   would let us run the image encoder on ANE in parallel with model
   load on the GPU. Test that ANE output matches MLX output within
   tolerance.

GUIDANCE
- Verify against the answers in docs/open-questions-resolved.md for
  DINOv3 input size and which layer's features TRELLIS.2 actually uses.
  Don't guess — those answers are why Phase 1 exists.
- The PyTorch reference test is non-negotiable. It catches RoPE base
  frequency mistakes, normalization order mistakes, and weight-shape
  transposition issues that would otherwise silently corrupt
  generation later.
- Keep dinov3.py self-contained. No imports from other trellis2_mlx
  modules.

WHEN TO STOP AND ASK
- Activation mismatch between PyTorch and MLX exceeds tolerance after
  you've eliminated obvious bugs (weight transpose, normalization eps,
  positional embedding interpolation).
- CoreML conversion produces output that diverges from MLX — this
  usually means an op was unsupported and silently fell back; we need
  to know.

End-of-session summary: max per-layer error vs PyTorch reference,
output shape, any deviations from the spec.

╚══════════════════════════════════════════════════════════════════════

**Done when:** test_dinov3.py passes with the documented tolerance,
demo script runs and produces sensible output, weight_convert.py has
been used to load the DINOv3 weights and is generic enough to reuse.

---

# Phase 3: O-Voxel Data Layer + Submanifold Sparse Conv

**Goal:** Build the foundational data structure and the most-used custom
op. This is the first phase that writes Metal kernels.

**Prerequisite:** Phase 2 done. Weight converter pattern established.

╔═══════════════════════════ PHASE 3 PROMPT ═══════════════════════════

Continuing trellis2-mlx. Phases 1-2 complete (skeleton, DINOv3-L in MLX
with weight converter pattern). This phase builds the O-Voxel data
structure and the most-used custom Metal kernel: submanifold sparse 3D
convolution. Read PHASE0_SPEC.md §3 (O-Voxel), §5.2 (sparse conv), and
§5.3 (neighbor table) before starting.

DELIVERABLES
1. trellis2_mlx/ovoxel/data.py — OVoxel dataclass per spec §3.2.
   Structure-of-arrays layout in MLX. Include from_dense() and to_dense()
   helpers for testing.
2. trellis2_mlx/metal/kernels/neighbor_build.metal — parallel
   open-addressing hashmap construction. Inputs: coords [L, 3] int32.
   Outputs: hash table (size 2L) and neighbor table N [L, 27] int32 with
   -1 for missing neighbors. Use atomic_compare_exchange_weak for linear
   probing. Per spec §5.3.
3. trellis2_mlx/metal/kernels/prefix_sum.metal — exclusive scan over
   uint32 array, used for compaction. Standard Blelloch scan or hybrid.
4. trellis2_mlx/metal/kernels/sparse_conv_fwd.metal — Masked Implicit
   GEMM submanifold conv per spec §5.2. Inputs: X [L, C_in], W
   [27, C_in, C_out], neighbor table N [L, 27]. Output: Y [L, C_out].
   Backward kernel deferred to Phase 7.
5. trellis2_mlx/metal/ops.py — MLX custom-op bindings for the above
   kernels. Forward only for now; we add vjp in Phase 7.
6. trellis2_mlx/nn/sparse_conv.py — SubMConv3 module wrapping the
   kernel. Includes weight initialization (Kaiming) and parameter
   registration.
7. Tests:
   - test_ovoxel.py — dense ↔ sparse roundtrip on synthetic data
   - test_neighbor_build.py — verify neighbor table matches a CPU
     reference implementation (Python dict lookup) for 1K, 10K, 100K
     active voxels
   - test_sparse_conv.py — numerical comparison vs PyTorch spconv on
     CPU (or a slow pure-PyTorch reference if spconv won't install
     CPU-only). Tolerance: 1e-3 fp32, 5e-3 bf16.

GUIDANCE
- Threadgroup memory budget on M4 Max: 32KB per threadgroup [verify
  with `system_profiler SPDisplaysDataType` or Apple docs]. Plan tile
  sizes accordingly.
- For the hashmap, use load factor 0.5 (table size = 2L). Linear probing
  is fine — there's no reason to use quadratic or double hashing for
  voxel grids.
- For implicit GEMM: don't allocate the gathered [L, 27, C_in] tensor.
  The whole point is to gather inside the matmul.
- Profile with Xcode's Metal frame capture once the kernels are
  correct. Note occupancy and memory bandwidth utilization in a
  docs/perf-notes.md file as you go.

WHEN TO STOP AND ASK
- Hashmap correctness fails — usually means atomic ordering or memory
  fences are wrong. This is subtle; show me the kernel.
- Sparse conv numerical mismatch vs reference after you've checked
  weight layout, neighbor table correctness, and accumulator dtype.
- M4 Max threadgroup memory turns out to be smaller than 32KB and tiles
  don't fit.

End-of-session summary: kernel correctness status, perf measurements
(GB/s achieved, % of peak memory bandwidth), any architectural surprises.

╚══════════════════════════════════════════════════════════════════════

**Done when:** All three tests pass, both Metal kernels are correct,
documented perf in `docs/perf-notes.md`.

---

# Phase 4: SC-VAE Decoder End-to-End Smoke Test

**Goal:** Wire the VAE decoder end-to-end with mesh extraction. First
visible deliverable — random latents in, GLB out. Generation quality
won't be good (it's random noise), but the pipeline must run cleanly.

**Prerequisite:** Phase 3 done.

╔═══════════════════════════ PHASE 4 PROMPT ═══════════════════════════

Continuing trellis2-mlx. Phases 1-3 complete. This phase wires the
SC-VAE decoder end-to-end with mesh extraction. Read PHASE0_SPEC.md §4.2
(VAE encoder for reference), §4.3 (VAE decoder — what we're building),
§5.4 (residual AE up/down), §5.5 (early pruning), §5.6 (mesh extraction),
§5.7 (material bake) before starting.

DELIVERABLES
1. trellis2_mlx/nn/res_ae.py — ResEnc and ResDec layers per spec §5.4.
   The "channel-to-space" (unstack + dup_groups) and "space-to-channel"
   (stack 8 children + avg_groups) shortcuts. Pure MLX, no Metal kernel
   needed (just reshape/gather/reduce).
2. trellis2_mlx/nn/pruning.py — early-pruning predictor head + compaction
   per spec §5.5. Uses the prefix_sum kernel from Phase 3.
3. trellis2_mlx/models/vae.py — SC-VAE decoder. Mirrors the encoder
   table in spec §4.2 with inverted stage order. Two instances: shape
   decoder and material decoder. Shape decoder produces (v, δ, γ);
   material decoder produces (c, m, r, α) and is conditioned on the
   shape decoder's pruning structure.
4. trellis2_mlx/metal/kernels/mesh_extract.metal — Flexible Dual Grid
   to triangle mesh per spec §5.6. One thread per voxel, iterates over
   3 axes, appends faces via atomic counter into pre-allocated face
   buffer (size 6L).
5. trellis2_mlx/metal/kernels/trilinear_bake.metal — material trilinear
   interpolation per spec §5.7. One thread per query point.
6. trellis2_mlx/ovoxel/mesh_extract.py — Python wrapper that calls the
   Metal kernel and returns vertex/face arrays.
7. trellis2_mlx/ovoxel/material_bake.py — Python wrapper for the bake
   kernel; produces per-vertex (c, m, r, α).
8. trellis2_mlx/ovoxel/postprocess.py — to_glb() function using trimesh.
   Packs base color → vertex colors; metallic/roughness/opacity into a
   custom GLB extension or material extras (whichever upstream uses;
   verify in reference/microsoft-trellis2/o-voxel/postprocess.py).
9. scripts/smoke_test_vae.py — Load pretrained SC-VAE shape + material
   decoders. Generate a random latent z ∈ [L_latent, 32] with L_latent
   ≈ 100. Decode through both VAEs. Run mesh extraction. Export as
   smoke_test.glb. Open in QuickLook (Cmd+Y in Finder) to verify it's
   a valid mesh — won't look like anything but should be topologically
   sound.
10. tests/test_mesh_extract.py — verify face count and winding order on
    synthetic O-Voxel inputs with known expected output.

GUIDANCE
- The mesh-extract winding order MUST match upstream. Verify against
  reference/microsoft-trellis2/o-voxel/. Reversed winding shows up as
  invisible / inside-out geometry in Unity / Blender.
- For material bake: missing voxels — verify upstream behavior in
  o-voxel/ source. Likely "nearest extrapolation" but verify.
- Don't bother optimizing perf in this phase. Correctness > speed.
  Phase 8 is for optimization.

WHEN TO STOP AND ASK
- Mesh has visible tears, inside-out faces, or non-manifold edges that
  shouldn't exist.
- VAE decoder output has NaN/inf — usually means an upsampling layer
  is missing the pruning mask or a residual shortcut.
- GLB doesn't open in QuickLook — file format issue, not algorithm.

End-of-session summary: smoke_test.glb file size, vertex count, face
count, whether it opens cleanly in QuickLook and Blender.

╚══════════════════════════════════════════════════════════════════════

**Done when:** A random-latent smoke test produces a valid GLB that
opens in QuickLook and Blender without errors. Don't worry about how
it looks.

---

# Phase 5: Sparse Attention (The Big One)

**Goal:** Write the flash-attention-style Metal kernel that dominates
inference time. This phase is the longest and has the highest perf impact.

**Prerequisite:** Phase 4 done.

╔═══════════════════════════ PHASE 5 PROMPT ═══════════════════════════

Continuing trellis2-mlx. Phases 1-4 complete. This phase implements the
biggest performance lever: fused sparse attention as a flash-attention-
style Metal kernel. Read PHASE0_SPEC.md §5.1 carefully. Also read the
FlashAttention-2 paper (arxiv 2307.08691) if you haven't.

DELIVERABLES
1. trellis2_mlx/metal/kernels/sparse_attn_fwd.metal — fused
   self-attention with tiled QK^T, online softmax, accumulated V.
   Inputs: Q, K, V each [L, H, D] (H=12, D=128, L up to ~9.6K).
   Optional mask. QK-Norm and RoPE applied OUTSIDE the kernel (in MLX
   ops feeding into it). Output: O [L, H, D]. Tile sizes target M4 Max
   threadgroup memory; aim for Br=64, Bc=64 (each 16KB in bf16, 32KB
   total fits the budget).
2. trellis2_mlx/nn/rope.py — 3D Rotary Position Embedding per the
   formulation verified in Phase 1. Apply to Q and K before the kernel
   call. Pure MLX.
3. trellis2_mlx/nn/sparse_attn.py — module wrapping the kernel:
   - Input projection (3 × linear)
   - QK-Norm (RMSNorm on Q and K with ε=1e-6)
   - RoPE 3D
   - Kernel call
   - Output projection
4. tests/test_sparse_attn.py — numerical comparison vs a reference
   implementation. Two references:
   - PyTorch CPU scaled_dot_product_attention with manual QK-Norm and
     RoPE — tolerance 5e-3 bf16
   - A pure-MLX (slow) implementation using mx.softmax on the full
     attention matrix — tolerance 1e-4 fp32
5. scripts/bench_sparse_attn.py — measure wall-clock and throughput for
   L ∈ {1K, 4K, 9.6K, 20K} at H=12, D=128. Output table + GB/s.
6. CrossAttention variant (sparse_attn.py): Q from voxel stream, K/V
   from image features. Same kernel, just call with different inputs;
   add the wrapper module.

GUIDANCE
- ONLINE SOFTMAX is the trick. Don't materialize the full L×L attention
  matrix. Track running max m and running sum l per Q-tile, rescale
  accumulator when m updates. The FA-2 paper has the equations.
- Use `simd_shuffle` for warp-level reductions in the softmax. This is
  critical for perf; without it the kernel is 2-3× slower.
- bf16 inputs and outputs, fp32 accumulators. The accumulator is what
  prevents catastrophic loss in long-sequence softmax.
- For the backward (deferred to Phase 7): note that we'll need to store
  the running sum l per Q-tile during forward. Pre-plan this in the
  kernel's output struct so we don't have to redesign.

WHEN TO STOP AND ASK
- Numerical error vs reference exceeds tolerance after you've checked:
  online softmax invariants (m, l updates), accumulator precision, RoPE
  application order, head dim handling.
- Kernel works for small L but produces NaN/inf for large L — usually
  threadgroup memory aliasing or out-of-bounds reads.
- Perf at L=9.6K is worse than 50ms — something is dispatch-overhead-
  bound or memory-bandwidth-pathological; we need to investigate
  together with Xcode frame capture.

End-of-session summary: tolerance achieved vs reference, ms at each L,
GB/s and percentage of M4 Max peak (~400-540 GB/s memory bandwidth).

╚══════════════════════════════════════════════════════════════════════

**Done when:** Forward kernel matches references within tolerance,
benchmarks show ≥50% of theoretical memory bandwidth at L=9.6K.

---

# Phase 6: DiT Block + Full Three-Stage Pipeline

**Goal:** Compose attention + MLP + AdaLN into the DiT block, wire all
three generator stages, and produce the first real image-to-3D output.

**Prerequisite:** Phase 5 done.

╔═══════════════════════════ PHASE 6 PROMPT ═══════════════════════════

Continuing trellis2-mlx. Phases 1-5 complete. This phase assembles the
DiT block, all three DiT generator stages, and the rectified-flow
sampler. End of this phase: image-to-3D works end-to-end. Read
PHASE0_SPEC.md §2 (pipeline), §4.4 (DiT block + AdaLN-single + QK-Norm),
§2.1 (sampling).

DELIVERABLES
1. trellis2_mlx/nn/adaln.py — AdaLN-single per PixArt-α. Shared MLP
   predicts modulation params from timestep embedding; per-layer learned
   adapters scale them. Two modulation branches per DiT block (one for
   attn, one for FFN).
2. trellis2_mlx/nn/dit_block.py — DiT block per spec §4.4: self-attn
   (with QK-Norm + RoPE), cross-attn (no RoPE), FFN (1536 → 8192 →
   1536). AdaLN-single modulation on attn and FFN.
3. trellis2_mlx/models/dit.py — three model classes:
   - SparseStructureDiT (stage 1, operates on dense N³ grid)
   - GeometryDiT (stage 2, sparse, in_dim=32)
   - MaterialDiT (stage 3, sparse, in_dim=64 with shape concat)
   All share the DiT block; only the input projection and active-set
   handling differ.
4. trellis2_mlx/samplers/rectified_flow.py — rectified-flow sampler
   with classifier-free guidance. Configurable step count and CFG
   scale per stage (defaults from upstream — verified in Phase 1).
5. trellis2_mlx/pipeline.py — flesh out Trellis2ImageTo3DPipeline:
   from_pretrained() loads all three DiTs + both VAE decoders + DINOv3
   weights, configures device. run(image) executes the 9 pipeline
   stages from spec §2 and returns an OVoxelMesh.
6. trellis2_mlx/utils/preprocess.py — image preprocessing: RMBG-2.0
   background removal, center-crop, resize, normalize. Verify exact
   transform matches upstream pipelines/image_to_3d.py.
7. scripts/run_image_to_3d.py — CLI: takes an image path, runs the
   pipeline, exports GLB. Add a --resolution flag for 512/1024.
8. tests/test_pipeline_e2e.py — runs the full pipeline on a stock test
   image (assets/example_image/T.png from upstream). Doesn't check
   visual quality but verifies: GLB is valid, vertex count is
   reasonable (>10K, <2M), no NaN in any intermediate latent.

GUIDANCE
- The three DiTs share architecture but NOT weights. Each has its own
  checkpoint subdirectory.
- Stage 1 operates on a DENSE small grid (32³ or 64³ voxels). Stage 2
  and 3 operate on the SPARSE active set from stage 1. Stage 2's output
  determines the active set used in stage 3.
- CFG implementation: classifier-free guidance scale per stage,
  guidance applied via the standard formula
  v_guided = v_uncond + scale * (v_cond - v_uncond).
- bf16 throughout. fp32 for the rectified-flow integrator's t variable
  and for AdaLN-single's shared MLP if numerical stability is an issue.

WHEN TO STOP AND ASK
- End-to-end output is degenerate (empty mesh, all-voxel-active blob,
  NaN latents). This usually means RoPE base or QK-Norm placement is
  wrong — both are common failure modes.
- Stage 1 produces an unreasonable number of active voxels (much more
  or fewer than expected ~9.6K at 1024³).
- Cross-attention image conditioning seems to be ignored — output
  doesn't change when the input image changes.

End-of-session summary: wall-clock per stage at 512³ and 1024³, final
GLB visual quality vs upstream reference on the same input image,
remaining issues.

╚══════════════════════════════════════════════════════════════════════

**Done when:** `run_image_to_3d.py assets/example_image/T.png` produces
a recognizable GLB that visually resembles the upstream output for the
same input.

---

# Phase 7: Backward Kernels for Fine-Tuning

**Goal:** Add vjp (backward) for every custom op so fine-tuning works.
This is the gate for the "fine-tuning later" promise.

**Prerequisite:** Phase 6 done. Inference works end-to-end.

╔═══════════════════════════ PHASE 7 PROMPT ═══════════════════════════

Continuing trellis2-mlx. Phases 1-6 complete; inference works end-to-end.
This phase adds backward passes to every custom Metal kernel so we can
fine-tune. Read PHASE0_SPEC.md §5 (kernels with backward specs) and skim
the FlashAttention-2 backward section.

DELIVERABLES
1. trellis2_mlx/metal/kernels/sparse_conv_bwd.metal — backward for
   submanifold sparse conv:
   - dX: scatter-add of dY @ W_k^T across neighbor positions
   - dW: per kernel position k, gather X rows and outer-product with dY
2. trellis2_mlx/metal/kernels/sparse_attn_bwd.metal — flash-attention
   backward. Uses the saved running sum from forward (planned in Phase
   5). Two-pass: compute dV and dP first, then dQ and dK from the
   recomputed S.
3. Update trellis2_mlx/metal/ops.py to register vjp rules for both ops
   via the MLX C++ extension interface (or whatever the current MLX API
   is for custom autograd — verify against MLX version pinned in
   pyproject).
4. tests/test_backward.py — gradient correctness via finite differences
   on small inputs. For each custom op:
   - Sample random input
   - Run forward, then backward to get analytical gradient
   - Compute numerical gradient via central differences
   - Verify they match (tolerance ~1e-2 in bf16, ~1e-4 in fp32)
5. scripts/finetune_demo.py — minimal fine-tuning loop on a tiny
   synthetic dataset (10 image+mesh pairs):
   - Load pretrained DiTs
   - Freeze VAE and DINOv3
   - Run rectified-flow training loop for 100 steps on stage 2
     (geometry DiT) only
   - Verify loss decreases monotonically (or at least trends down)
   - Save fine-tuned checkpoint
6. docs/finetuning.md — how to prepare a dataset, what's frozen vs
   trainable, expected memory usage, throughput.

GUIDANCE
- Backward for sparse attention is roughly 2× the forward FLOPs. Plan
  threadgroup memory accordingly.
- Scatter-add in dX backward needs atomics. Use atomic_fetch_add_explicit
  on fp32 (Apple GPUs support fp32 atomics; bf16 atomics need software
  emulation — accumulate in fp32 and cast at end).
- Memory budget for fine-tuning: full optimizer state for one DiT (1.3B
  params × 16 bytes Adam = 20GB). Plus activations. Stage 2 only is
  doable on 48GB+ M4 Max; full pipeline training needs gradient
  checkpointing or LoRA.

WHEN TO STOP AND ASK
- Finite-difference test fails. Usually means a sign error or missing
  factor (1/√d in attention, for example) in the backward.
- Fine-tune loss is flat or rising — gradient is wrong or LR is wildly
  off; investigate together.
- OOM during fine-tune even on stage 2 only — need to discuss gradient
  checkpointing strategy.

End-of-session summary: backward correctness per kernel, fine-tune loss
curve over 100 steps, memory peak during fine-tune.

╚══════════════════════════════════════════════════════════════════════

**Done when:** Backward kernels pass finite-difference tests,
finetune_demo.py shows loss decreasing on synthetic data.

---

# Phase 8: Optimization & Profiling

**Goal:** Hunt down inefficiencies. Saturate the M4 Max GPU.

**Prerequisite:** Phase 7 done. Everything correct, but probably not as
fast as it could be.

╔═══════════════════════════ PHASE 8 PROMPT ═══════════════════════════

Continuing trellis2-mlx. Phases 1-7 complete: full pipeline works,
backward kernels exist. This phase is pure optimization. Target
performance per PHASE0_SPEC.md §1.2: 512³ in ≤45s, 1024³ in ≤180s on
M4 Max.

DELIVERABLES
1. Full Xcode Metal frame capture analysis. Document in docs/perf-analysis.md
   for each kernel:
   - Wall-clock per invocation
   - GPU occupancy (active warps / max)
   - Memory bandwidth achieved (GB/s) vs M4 Max peak (~400-540 GB/s)
   - Register pressure (spills?)
   - Threadgroup memory usage
2. Optimization pass on each kernel based on profiler findings.
   Likely candidates:
   - Sparse attention: tune tile sizes for M4 Max specifically; try
     Br ∈ {32, 64, 96, 128} and pick winner.
   - Sparse conv: try fused bias + activation in the same kernel.
   - Mesh extract: parallelize over (voxel, axis) pairs instead of
     just voxels.
   - Neighbor build: try cuckoo hashing or robin-hood hashing if linear
     probing has high probe counts on real data.
3. End-to-end pipeline timing pass:
   - Identify any CPU↔GPU sync points and eliminate where possible
   - Overlap DINOv3 encoding with model load (or move DINOv3 to ANE
     via CoreML from Phase 2 optional)
   - Fuse adjacent elementwise ops in MLX (often picked up automatically
     by lazy eval but worth verifying)
4. Memory optimization:
   - Profile peak unified memory usage at 1024³
   - Apply MLX gradient checkpointing in fine-tune path
   - Consider kv-cache reuse across rectified-flow steps (if applicable)
5. scripts/bench_full_pipeline.py — runs the full image-to-3D pipeline
   on a fixed test image 5 times, reports min/median/max wall-clock at
   512³ and 1024³. Compare against the trellis-mac PyTorch MPS port on
   the same hardware (≈3.5min on M4 Pro 24GB) — we should be
   significantly faster.
6. docs/benchmarks.md — formal benchmark numbers on M4 Max with
   configuration details (chip variant, RAM, macOS version, MLX
   version, model resolution).

GUIDANCE
- Set hard targets and don't ship until they're hit:
  - Sparse attention at L=9.6K, H=12, D=128: <50ms
  - End-to-end 512³: <45s
  - End-to-end 1024³: <180s
- Don't optimize for synthetic benchmarks. Optimize for real pipeline
  wall-clock.
- If a candidate optimization is complex and unproven, prototype on a
  branch and benchmark before merging.

WHEN TO STOP AND ASK
- Targets not hit after exhausting obvious wins. We may need to
  architecturally rework something (e.g., switch sparse attention to
  block-sparse with a coarser sparsity pattern at the cost of small
  quality regression).
- Profiler shows we're already at ≥80% of peak memory bandwidth — at
  that point further perf wins require algorithmic changes, not
  micro-optimization.

End-of-session summary: before/after timings, the optimization
hit-list and what landed, percent of theoretical peak bandwidth.

╚══════════════════════════════════════════════════════════════════════

**Done when:** Performance targets hit, benchmarks documented.

---

# Phase 9: Release

**Goal:** Polish, document, and ship a public release.

**Prerequisite:** Phase 8 done. Targets hit.

╔═══════════════════════════ PHASE 9 PROMPT ═══════════════════════════

Continuing trellis2-mlx. Phases 1-8 complete: full pipeline works at
target performance. This is the release phase. Polish everything for
public consumption.

DELIVERABLES
1. README.md overhaul:
   - Clear "what + why" intro
   - Hardware requirements with tested configurations
   - Quick-start: install, run on example image, expected output
   - Benchmarks table (from Phase 8)
   - Architecture overview (lifted from PHASE0_SPEC.md)
   - Comparison to trellis-mac (honest perf comparison)
   - Citation block (paper + this repo)
2. docs/installation.md — detailed install with all troubleshooting
   gathered during development (Xcode CLT, MLX version pins, etc.)
3. docs/architecture.md — distilled version of PHASE0_SPEC.md for
   contributors. Focus on the "why" of each design choice.
4. examples/ directory:
   - examples/quickstart.ipynb — runs the pipeline on an example image
   - examples/finetuning.ipynb — minimal LoRA-style fine-tune demo
   - examples/batch_inference.py — efficient batched generation script
5. CI setup (.github/workflows/ci.yml): ruff, mypy, pytest. macOS-arm64
   runners only.
6. Release v0.1.0:
   - Tag the commit
   - Write release notes summarizing what's in v0.1.0
   - gh release create v0.1.0
7. Upload converted MLX weights to Hugging Face as
   {user}/trellis2-mlx-4B — pretrained weights pre-converted from
   Microsoft's checkpoint, with our weight_convert.py reproducibility
   script.
8. Optional: announce. Hacker News "Show HN" post, /r/LocalLLaMA,
   /r/MachineLearning, twitter/X with benchmark screenshot.

GUIDANCE
- README is the front door. Spend real time on it.
- Quickstart must work for someone with zero context. Test it on a
  fresh checkout in a fresh venv.
- Don't oversell. Be honest about the H100 gap and the M4 Max ceiling.
- License compliance: TRELLIS.2 is MIT, DINOv3 has its own license
  (verify), nvdiffrec is not used (we skipped it). Spell out the
  dependency license inheritance clearly.

WHEN TO STOP AND ASK
- Quickstart breaks on a clean environment.
- HF weight upload fails or there are licensing questions about
  redistributing converted weights (check Microsoft's terms).

End-of-session summary: release URL, HF model URL, any deferred items
that need a v0.1.1.

╚══════════════════════════════════════════════════════════════════════

**Done when:** v0.1.0 is tagged and public, quickstart works from a
clean checkout, weights are on Hugging Face.

---

## Beyond Phase 9 (Optional)

Possible follow-ups that aren't on the critical path:

- **Phase 10: ANE offload.** Move DINOv3 to the Neural Engine via
  CoreML. Frees the GPU for the DiTs to run in parallel with image
  encoding. ~5-10% wall-clock improvement, modest complexity.

- **Phase 11: 1536³ cascaded inference.** Implement the cascaded
  upsampling trick from the paper (§4.5) to go above 1024³ without
  retraining. High value for game assets.

- **Phase 12: LoRA fine-tuning.** Low-rank adapter fine-tuning of the
  three DiTs. Drastically reduces memory pressure (no full optimizer
  state) and disk size for checkpoints. Standard implementation.

- **Phase 13: Unity Plugin.** Native Swift framework wrapper around the
  MLX library, exposed as a Unity plugin. One-click "import image,
  generate 3D asset" inside the Unity editor.

These are open-ended. Discuss scope with me before starting any of them.

---

## How to recover if a phase goes sideways

1. **Roll back the offending commits.** Conventional commits mean
   `git revert <sha>..HEAD` cleanly undoes a bad phase.
2. **Re-read the relevant spec section** in PHASE0_SPEC.md. Often a
   misimplementation traces to a misread of the spec, not a fundamental
   flaw in the plan.
3. **Check docs/open-questions-resolved.md.** If you guessed an answer
   to a [VERIFY] question that turned out wrong, fix the doc first,
   then the code.
4. **Ask me.** That's what I'm here for.
