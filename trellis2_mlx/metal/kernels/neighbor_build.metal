// Parallel open-addressing hashmap → neighbor table.
// Implements PHASE0_SPEC.md §5.3.
//
// Inputs:
//   coords ∈ [L, 3]  int32 voxel coordinates
// Outputs:
//   hash table (size 2L, load factor 0.5)
//   N ∈ [L, 27]      int32 neighbor table (-1 for missing)
//
// Kernel A: parallel insert with linear probing on conflict
//   (atomic_compare_exchange on the slot).
// Kernel B: per voxel × per kernel offset, hash-lookup coords[i] + offset[k]
//   and write the resulting index or -1.
//
// Forward-only — index structure, not differentiable.

#include <metal_stdlib>
using namespace metal;

// TODO(phase1-step4): implement.
