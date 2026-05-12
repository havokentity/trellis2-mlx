// Flexible Dual Grid → triangle mesh extraction.
// Implements PHASE0_SPEC.md §5.6.
//
// One thread per voxel × 3 axes:
//   - If δ_i[axis] is not set, skip.
//   - Otherwise look up the 4 voxels around the edge via the spatial hash.
//   - If any of the 4 is missing, skip.
//   - Adaptively split the quad into 2 triangles using γ.
//   - Atomic-append into the output face buffer.
//
// Pre-allocate F with worst-case size 6L (3 axes × 2 triangles per face).
// Quad winding-order convention is open question §8 Q7 — see
// docs/open-questions-resolved.md.

#include <metal_stdlib>
using namespace metal;

// TODO(phase1-step6): implement.
