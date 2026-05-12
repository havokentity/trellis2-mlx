// Submanifold 3×3×3 sparse convolution — BACKWARD.
// Implements PHASE0_SPEC.md §5.2 (backward section).
//
// dX:  scatter-add dY @ W_k^T contributions back to neighbor positions,
//      using the neighbor table to index destinations.
// dW:  per kernel offset k, sum outer products of gathered X rows with dY rows.
//
// Required because fine-tuning is in scope (CLAUDE.md custom-op policy).
// fp32 accumulators throughout.

#include <metal_stdlib>
using namespace metal;

// TODO(phase1-step11): implement.
