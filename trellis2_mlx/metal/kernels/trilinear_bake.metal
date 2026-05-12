// Trilinear material baking at query points.
// Implements PHASE0_SPEC.md §5.7.
//
// One thread per query point:
//   - Compute the 8 surrounding voxel coords.
//   - Look up each via the spatial hash.
//   - Trilinear-interpolate (c, m, r, α). Behavior for missing voxels
//     (nearest extrapolation vs zero) is open question — see
//     docs/open-questions-resolved.md.

#include <metal_stdlib>
using namespace metal;

// TODO(phase1-step10): implement.
