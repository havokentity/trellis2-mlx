// Exclusive parallel prefix-sum over a flat mask.
// Implements PHASE0_SPEC.md §5.5 (compaction support).
//
// Used after the early-pruning predictor: given a child-survival mask
// ρ ∈ {0,1}^(L_coarse × 8), compute exclusive scan to derive compacted
// output indices, then scatter the surviving children into a tight
// [L_fine, ...] tensor.
//
// Standard Blelloch up-sweep/down-sweep on threadgroup tiles.

#include <metal_stdlib>
using namespace metal;

// TODO(phase1-step6): implement.
