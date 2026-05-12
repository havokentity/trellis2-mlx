// Flash-attention-style sparse-token self-attention — BACKWARD.
// Implements PHASE0_SPEC.md §5.1 (backward section).
//
// Standard flash-attention backward: recompute attention in the forward
// direction, then accumulate gradients in two passes — one for dQ, one for
// dK/dV. Roughly 2× the forward cost.
//
// Required because fine-tuning is in scope (CLAUDE.md custom-op policy).
// fp32 accumulators throughout.

#include <metal_stdlib>
using namespace metal;

// TODO(phase1-step11): implement.
