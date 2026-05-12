// Flash-attention-style sparse-token self-attention — FORWARD.
// Implements PHASE0_SPEC.md §5.1.
//
// Inputs:
//   Q, K, V ∈ [L, H, D]  (bf16; H=12, D=128, L up to ~9.6K)
//   optional mask
// Output:
//   O ∈ [L, H, D]
//
// Algorithm: tiled online-softmax.
//   - Q tile: 64 × 128 bf16 = 16 KB threadgroup memory
//   - K tile: 64 × 128 bf16 = 16 KB threadgroup memory
//   - Total 32 KB — sits at the M4 threadgroup limit [VERIFY in spec §5.1].
//   - Per Q-tile: load Q, iterate K/V tiles, online-softmax update (max, sum),
//     accumulate weighted V in fp32 accumulator, write O on completion.
//
// fp32 for the running max, sum, and output accumulator (CLAUDE.md dtype rule).
// QK-Norm and RoPE-3D are applied by the caller before this kernel runs.

#include <metal_stdlib>
using namespace metal;

// TODO(phase1-step7): implement.
