// Submanifold 3×3×3 sparse convolution — FORWARD.
// Implements PHASE0_SPEC.md §5.2 (Masked Implicit GEMM).
//
// Inputs:
//   X ∈ [L, Cin]                voxel features (bf16)
//   N ∈ [L, 27]                 neighbor table (int32, -1 for missing)
//   W ∈ [27, Cin, Cout]         conv weights (bf16)
//   bias ∈ [Cout] (optional)
// Output:
//   Y ∈ [L, Cout]               (bf16)
//
// Strategy: fuse the 27 partial GEMMs into one large implicit-GEMM by treating
// the neighbor table as a permutation of the input rows. Missing neighbors
// (-1) are handled via predicated stores. fp32 accumulator.

#include <metal_stdlib>
using namespace metal;

// TODO(phase1-step5): implement.
